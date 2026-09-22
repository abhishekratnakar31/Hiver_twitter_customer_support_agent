"""
src/models/reply_generator.py

Phase 5 Top-1 Historical Response Baseline:
Module hygiene note: Contains baseline response strategies, including a retrieval-copy baseline. Does not use an LLM.
Retrieves nearest historical resolution from train.csv and returns the exact historical response without modification.
Computes secondary exploratory surface text metrics (BLEU-1/2/4, ROUGE-1/2/L) against historical_reference_reply.
(BLEU/ROUGE are reported as secondary diagnostics only, and are not interpreted as measures of correctness or quality).
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def compute_token_ngram_overlap(hyp_tokens, ref_tokens, n=1):
    """Calculates n-gram precision for BLEU diagnostic calculation."""
    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0
    
    hyp_ngrams = [tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)]
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
    
    from collections import Counter
    hyp_counts = Counter(hyp_ngrams)
    ref_counts = Counter(ref_ngrams)
    
    overlap = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
    return overlap / len(hyp_ngrams)


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Computes secondary exploratory BLEU-1, BLEU-2, BLEU-4 scores."""
    b1_scores, b2_scores, b4_scores = [], [], []
    
    for hyp, ref in zip(hypotheses, references):
        hyp_toks = str(hyp).lower().split()
        ref_toks = str(ref).lower().split()
        
        p1 = compute_token_ngram_overlap(hyp_toks, ref_toks, n=1)
        p2 = compute_token_ngram_overlap(hyp_toks, ref_toks, n=2)
        p4 = compute_token_ngram_overlap(hyp_toks, ref_toks, n=4)
        
        # Brevity penalty
        l_hyp = len(hyp_toks)
        l_ref = len(ref_toks)
        bp = 1.0 if l_hyp > l_ref else (np.exp(1 - l_ref / max(1, l_hyp)) if l_hyp > 0 else 0.0)
        
        b1_scores.append(bp * p1)
        b2_scores.append(bp * np.sqrt(p1 * p2))
        b4_scores.append(bp * (p1 * p2 * p4) ** 0.25 if (p1 * p2 * p4) > 0 else 0.0)
        
    return {
        "bleu_1": float(np.mean(b1_scores)) if b1_scores else 0.0,
        "bleu_2": float(np.mean(b2_scores)) if b2_scores else 0.0,
        "bleu_4": float(np.mean(b4_scores)) if b4_scores else 0.0,
    }


def compute_rouge_l(hyp_toks, ref_toks):
    """Computes ROUGE-L LCS score."""
    m = len(hyp_toks)
    n = len(ref_toks)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_toks[i-1] == ref_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    rec = lcs_len / n
    prec = lcs_len / m
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Computes secondary exploratory ROUGE-1, ROUGE-2, ROUGE-L scores."""
    r1_f1s, r2_f1s, rl_f1s = [], [], []
    
    for hyp, ref in zip(hypotheses, references):
        hyp_toks = str(hyp).lower().split()
        ref_toks = str(ref).lower().split()
        
        p1 = compute_token_ngram_overlap(hyp_toks, ref_toks, n=1)
        r1 = compute_token_ngram_overlap(ref_toks, hyp_toks, n=1)
        f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0
        
        p2 = compute_token_ngram_overlap(hyp_toks, ref_toks, n=2)
        r2 = compute_token_ngram_overlap(ref_toks, hyp_toks, n=2)
        f1_2 = (2 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0
        
        _, _, f1_l = compute_rouge_l(hyp_toks, ref_toks)
        
        r1_f1s.append(f1_1)
        r2_f1s.append(f1_2)
        rl_f1s.append(f1_l)
        
    return {
        "rouge_1": float(np.mean(r1_f1s)) if r1_f1s else 0.0,
        "rouge_2": float(np.mean(r2_f1s)) if r2_f1s else 0.0,
        "rouge_l": float(np.mean(rl_f1s)) if rl_f1s else 0.0,
    }


class Top1HistoricalResponseBaseline:
    """
    Top-1 Historical Response Baseline.
    Retrieves nearest historical resolution from train.csv and returns the exact historical response without modification.
    Does NOT use an LLM.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def generate_reply(self, query_text: str) -> Dict[str, Any]:
        """Retrieves Top-1 historical response for a single query."""
        results = self.retriever.retrieve(query_text, k=1)
        if not results:
            return {
                "generated_reply": "",
                "evidence_conversation_id": None,
                "evidence_interaction_id": None,
                "similarity_score": 0.0
            }
        
        top1 = results[0]
        return {
            "generated_reply": top1["brand_response"],
            "evidence_conversation_id": top1["conversation_id"],
            "evidence_interaction_id": top1["interaction_id"],
            "similarity_score": top1["similarity_score"]
        }

    def generate_reply_batch(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        """Retrieves Top-1 historical response for a batch of queries."""
        batch_results = self.retriever.retrieve_batch(query_texts, k=1)
        outputs = []
        for i, res in enumerate(batch_results):
            if not res:
                outputs.append({
                    "generated_reply": "",
                    "evidence_conversation_id": None,
                    "evidence_interaction_id": None,
                    "similarity_score": 0.0
                })
            else:
                top1 = res[0]
                outputs.append({
                    "generated_reply": top1["brand_response"],
                    "evidence_conversation_id": top1["conversation_id"],
                    "evidence_interaction_id": top1["interaction_id"],
                    "similarity_score": top1["similarity_score"]
                })
        return outputs

    def evaluate_secondary_diagnostics(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """Calculates secondary exploratory BLEU and ROUGE surface text metrics."""
        bleu_metrics = compute_bleu(hypotheses, references)
        rouge_metrics = compute_rouge(hypotheses, references)
        
        combined = {}
        combined.update(bleu_metrics)
        combined.update(rouge_metrics)
        return combined


class GroundedLLMGenerator:
    """
    Phase 6 Grounded LLM Response Generator.
    Generates empathetic Twitter support responses strictly conditioned on Top-K retrieved historical resolutions.
    
    System Prompt Guidelines:
    - Treats historical retrieved responses as handling evidence, NOT authoritative policy truth.
    - Does NOT infer unverified guarantees, refunds, timelines, or binding commitments.
    - Maintains a polite, professional brand voice.
    """

    SYSTEM_PROMPT = """You are an AI customer support representative for AmazonHelp on Twitter.
Your goal is to provide a helpful, polite, and empathetic response to the customer.

IMPORTANT GROUNDING RULES:
1. Use the provided retrieved historical cases strictly as EXAMPLES of prior handling patterns.
2. Historical responses are evidence of handling patterns, NOT authoritative corporate policy truth.
3. Do NOT invent or infer unverified policies, refund guarantees, account access changes, or specific delivery promises that are not supported by the evidence.
4. If the issue requires account verification or private details, advise the customer to reach out via Direct Message (DM) with relevant details (e.g., order number).
5. Keep the response concise, clear, and under 280 characters suitable for Twitter.
"""

    def __init__(self, mock_mode: bool = False, model_name: str = "gemini-2.5-flash"):
        import os
        from pathlib import Path
        for p in [Path.home() / ".env", Path(".env")]:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

        self.mock_mode = mock_mode or os.environ.get("MOCK_LLM", "0") == "1"
        self.model_name = model_name

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates a grounded support reply given customer message, predicted intent, and retrieved evidence.
        """
        # Format evidence context
        evidence_snippets = []
        evidence_ids = []
        for idx, ev in enumerate(retrieved_evidence, 1):
            evidence_snippets.append(
                f"Example {idx} (Similarity: {ev.get('similarity_score', 0.0):.2f}):\n"
                f"  Customer Inquiry: {ev.get('customer_query', '')}\n"
                f"  Historical Reply: {ev.get('brand_response', '')}"
            )
            evidence_ids.append(ev.get('interaction_id', ''))

        context_block = "\n\n".join(evidence_snippets)

        prompt = (
            f"Customer Inquiry: \"{customer_message}\"\n"
            f"Predicted Intent Category: {predicted_intent}\n\n"
            f"Retrieved Historical Support Examples:\n{context_block}\n\n"
            f"Draft a grounded, polite Twitter response (<=280 chars):"
        )

        if self.mock_mode:
            # Deterministic compliant mock response for unit tests / offline dry-runs
            if "order" in customer_message.lower() or "track" in customer_message.lower():
                reply = "Please send us a DM with your order ID and email address so we can check your shipment status for you."
            elif "cancel" in customer_message.lower() or "refund" in customer_message.lower():
                reply = "We'd be glad to look into your cancellation request. Please DM us your order details so we can assist."
            elif "account" in customer_message.lower() or "password" in customer_message.lower():
                reply = "For security assistance with your account, please send us a Direct Message with your email."
            else:
                reply = "We are here to help! Please send us a Direct Message with your order details so we can investigate."

            return {
                "generated_reply": reply,
                "retrieved_evidence_ids": evidence_ids,
                "model_used": "mock_generator",
                "is_mock": True
            }

        # Real LLM API Call
        import os
        import json
        import urllib.request

        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "systemInstruction": {"parts": [{"text": self.SYSTEM_PROMPT}]}
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            
            # Retry loop for Gemini HTTP 429 Rate Limit
            reply = None
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        reply = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                        time.sleep(5.0 * (attempt + 1))
                    else:
                        print(f"Warning: Gemini API {e.code} error after retries. Using fallback reply.")
                        break
                except Exception as e:
                    print(f"Warning: Gemini API call error: {e}")
                    break
            
            return {
                "generated_reply": reply or "Please send us a DM with your order details so we can assist.",
                "retrieved_evidence_ids": evidence_ids,
                "model_used": "gemini-2.5-flash",
                "is_mock": False
            }
        elif openai_key:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            })
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                reply = res_data["choices"][0]["message"]["content"].strip()
            
            return {
                "generated_reply": reply,
                "retrieved_evidence_ids": evidence_ids,
                "model_used": "gpt-4o-mini",
                "is_mock": False
            }
        else:
            raise ValueError("Real LLM evaluation requires GEMINI_API_KEY or OPENAI_API_KEY in environment!")

