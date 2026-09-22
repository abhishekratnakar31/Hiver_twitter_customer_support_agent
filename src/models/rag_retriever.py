"""
src/models/rag_retriever.py

Phase 5 RAG Retrieval Module:
Dense vector retriever built over ~35,000 training customer queries in train.csv using FAISS IndexFlatIP
with sentence-transformers/all-MiniLM-L6-v2.
Reports Mean Cosine Similarity and Top-K similarity distributions as baseline retrieval diagnostics.
Restricted exclusively to train.csv (zero leakage into val, test pool, or golden set).
"""

import os
import json
import faiss
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """Dense vector retriever using FAISS over training interactions."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []  # List of dicts: interaction_id, conversation_id, customer_message, brand_response

    def build_index(self, train_df):
        """Builds FAISS index strictly from train.csv dataframe."""
        if train_df.empty:
            raise ValueError("train_df is empty! Cannot build vector index.")
        
        required_cols = ["interaction_id", "conversation_id", "customer_message", "brand_response"]
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing required column '{col}' in train_df")
        
        self.metadata = train_df[required_cols].to_dict(orient="records")
        queries = train_df["customer_message"].astype(str).tolist()
        
        # Compute normalized dense embeddings for inner product cosine similarity
        embeddings = self.model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"Built FAISS vector index with {self.index.ntotal:,} training vectors (dim={dimension}).")
        return self

    def retrieve(self, query_text, k=5):
        """Retrieves Top-K nearest historical resolution pairs for a single query text."""
        if self.index is None or len(self.metadata) == 0:
            raise RuntimeError("FAISS index is not built or loaded! Call build_index() or load() first.")
        
        q_emb = self.model.encode([str(query_text)], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)
        
        scores, indices = self.index.search(q_emb, k)
        results = []
        for rank in range(min(k, len(indices[0]))):
            idx = indices[0][rank]
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["similarity_score"] = float(scores[0][rank])
            item["rank"] = rank + 1
            results.append(item)
        return results

    def retrieve_batch(self, query_texts, k=5):
        """Retrieves Top-K nearest historical resolution pairs for a list of query texts."""
        if self.index is None or len(self.metadata) == 0:
            raise RuntimeError("FAISS index is not built or loaded! Call build_index() or load() first.")
        
        q_embs = self.model.encode(list(query_texts), normalize_embeddings=True, show_progress_bar=False)
        q_embs = np.array(q_embs, dtype=np.float32)
        
        scores, indices = self.index.search(q_embs, k)
        batch_results = []
        for i in range(len(query_texts)):
            item_results = []
            for rank in range(min(k, len(indices[i]))):
                idx = indices[i][rank]
                if idx < 0 or idx >= len(self.metadata):
                    continue
                item = dict(self.metadata[idx])
                item["similarity_score"] = float(scores[i][rank])
                item["rank"] = rank + 1
                item_results.append(item)
            batch_results.append(item_results)
        return batch_results

    def compute_similarity_diagnostics(self, query_texts, k=5):
        """Computes mean cosine similarity and Top-K score distributions across query texts."""
        batch_results = self.retrieve_batch(query_texts, k=k)
        top1_scores = [res[0]["similarity_score"] for res in batch_results if res]
        all_topk_scores = [[item["similarity_score"] for item in res] for res in batch_results if res]
        
        flat_topk = [s for res_scores in all_topk_scores for s in res_scores]
        
        diagnostics = {
            "num_queries": len(query_texts),
            "mean_top1_similarity": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "median_top1_similarity": float(np.median(top1_scores)) if top1_scores else 0.0,
            "min_top1_similarity": float(np.min(top1_scores)) if top1_scores else 0.0,
            "max_top1_similarity": float(np.max(top1_scores)) if top1_scores else 0.0,
            "mean_topk_similarity": float(np.mean(flat_topk)) if flat_topk else 0.0,
        }
        return diagnostics

    def save(self, index_path, metadata_path):
        """Saves FAISS index and metadata to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "metadata": self.metadata}, f, ensure_ascii=False)
        print(f"Saved FAISS index to {index_path} and metadata to {metadata_path}.")

    @classmethod
    def load(cls, index_path, metadata_path):
        """Loads FAISS index and metadata from disk."""
        if not Path(index_path).exists() or not Path(metadata_path).exists():
            raise FileNotFoundError(f"Missing index or metadata at {index_path}, {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_doc = json.load(f)
        
        instance = cls(model_name=meta_doc.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        instance.index = faiss.read_index(str(index_path))
        instance.metadata = meta_doc.get("metadata", [])
        print(f"Loaded FAISS vector index with {instance.index.ntotal:,} vectors.")
        return instance
