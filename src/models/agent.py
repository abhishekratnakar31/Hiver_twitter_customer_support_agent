"""
src/models/agent.py

Phase 6 End-to-End Support Agent Pipeline.
Orchestrates Intent Classification -> RAG Top-3 Retrieval -> Multi-Factor Escalation Policy -> Grounded LLM Response Generator.

Pre-Generation Escalation Precedence:
- Escalation triage occurs BEFORE response generation.
- If requires_human_escalation == True -> decision = "ESCALATE", returns reason, skips generation.
- If requires_human_escalation == False -> decision = "AUTO", calls GroundedLLMGenerator, returns reply.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from src.models.intent_classifier import TfidfLogisticClassifier
from src.models.rag_retriever import RAGRetriever
from src.models.reply_generator import GroundedLLMGenerator, Top1HistoricalResponseBaseline
from src.models.escalation_policy import BaselineEscalationPolicy
from src.models.conversation_memory import ConversationMemoryTracker


class TwitterSupportAgent:
    """
    Production-grade Twitter AI Support Agent Orchestrator.
    Supports multi-turn conversation memory with strict temporal anti-leakage.
    """

    def __init__(
        self,
        classifier: Optional[TfidfLogisticClassifier] = None,
        retriever: Optional[RAGRetriever] = None,
        generator: Optional[GroundedLLMGenerator] = None,
        policy: Optional[BaselineEscalationPolicy] = None,
        memory: Optional[ConversationMemoryTracker] = None
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator or GroundedLLMGenerator(mock_mode=True)
        self.policy = policy or BaselineEscalationPolicy(tau_conf=0.60, tau_sim=0.50)
        self.memory = memory or ConversationMemoryTracker(max_turns=4, max_context_chars=2000)

    @classmethod
    def load_from_models_dir(cls, models_dir: Union[str, Path] = "models", mock_llm: bool = False):
        """Loads pre-fitted model artifacts from models/ directory."""
        models_dir = Path(models_dir)
        clf_path = models_dir / "intent_tfidf_logreg.joblib"
        idx_path = models_dir / "rag_faiss.index"
        meta_path = models_dir / "rag_metadata.json"

        if not clf_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"Missing model artifacts in {models_dir}. Run scripts/train_baselines.py first!")

        classifier = TfidfLogisticClassifier.load(clf_path)
        retriever = RAGRetriever.load(idx_path, meta_path)
        generator = GroundedLLMGenerator(mock_mode=mock_llm)
        policy = BaselineEscalationPolicy(tau_conf=0.60, tau_sim=0.50)
        memory = ConversationMemoryTracker(max_turns=4, max_context_chars=2000)

        return cls(classifier=classifier, retriever=retriever, generator=generator, policy=policy, memory=memory)

    def process_inquiry(
        self,
        customer_message: str,
        interaction_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Processes a single customer inquiry end-to-end.
        Incorporate multi-turn context (preceding turns only) if conversation_id or conversation_history is provided.
        Emits structured JSON audit record.
        """
        start_time = time.time()
        interaction_id = interaction_id or f"INT_LIVE_{int(time.time()*1000)}"

        # Build augmented query context from preceding conversation history (temporal anti-leakage enforced)
        query_context = self.memory.get_full_context(
            conversation_id=conversation_id,
            current_message=customer_message,
            custom_history=conversation_history
        )

        # 1. Intent Classification (evaluated on query_context)
        intent_pred = self.classifier.predict([query_context])[0]
        probas = self.classifier.predict_proba([query_context])[0]
        max_conf = float(np.max(probas))

        # 2. RAG Retrieval (Top-3) (evaluated on query_context)
        retrieved_docs = self.retriever.retrieve(query_context, k=3)
        top1_sim = float(retrieved_docs[0]["similarity_score"]) if retrieved_docs else 0.0
        evidence_ids = [doc["interaction_id"] for doc in retrieved_docs]

        # 3. Escalation Policy Triage (Pre-Generation Check)
        triage_res = self.policy.evaluate_triage(
            predicted_intent=intent_pred,
            intent_confidence=max_conf,
            retrieval_similarity=top1_sim
        )

        requires_escalation = triage_res["requires_human_escalation"]
        escalation_reason = triage_res["escalation_reason"]

        if requires_escalation:
            # Pre-generation escalation: HALT generation and return audit log
            decision = "ESCALATE"
            generated_reply = None
            gen_metadata = {"status": "skipped_due_to_escalation"}
        else:
            # AUTO handling: Generate grounded response
            decision = "AUTO"
            gen_res = self.generator.generate_reply(
                customer_message=query_context,
                predicted_intent=intent_pred,
                retrieved_evidence=retrieved_docs
            )
            generated_reply = gen_res["generated_reply"]
            gen_metadata = gen_res
            if not getattr(self.generator, "mock_mode", True):
                time.sleep(4.2)

        # Update conversation memory state if conversation_id is provided
        if conversation_id:
            self.memory.add_turn(conversation_id, "Customer", customer_message)
            if generated_reply:
                self.memory.add_turn(conversation_id, "AmazonHelp", generated_reply)

        elapsed = time.time() - start_time

        return {
            "interaction_id": interaction_id,
            "conversation_id": conversation_id,
            "customer_message": customer_message,
            "query_context_used": query_context,
            "predicted_intent": intent_pred,
            "intent_confidence": round(max_conf, 4),
            "retrieval_top_k": len(retrieved_docs),
            "retrieval_similarity": round(top1_sim, 4),
            "requires_human_escalation": requires_escalation,
            "escalation_reason": escalation_reason,
            "decision": decision,
            "retrieved_evidence_ids": evidence_ids,
            "generated_response": generated_reply,
            "generation_metadata": gen_metadata,
            "elapsed_seconds": round(elapsed, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def process_batch(self, customer_messages: List[str]) -> List[Dict[str, Any]]:
        """Processes a batch of customer inquiries."""
        return [self.process_inquiry(msg) for msg in customer_messages]

