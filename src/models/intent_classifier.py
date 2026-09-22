"""
src/models/intent_classifier.py

Phase 5 Intent Classification Baselines:
1. MajorityClassifier (Baseline A): Predicts dominant intent class prior. No learning step.
2. TfidfLogisticClassifier (Baseline B): TF-IDF + Logistic Regression trained on train.csv with hyperparam selection on validation.csv.
3. ZeroShotEmbeddingSimilarityBaseline (Baseline C): Dense embedding cosine similarity (all-MiniLM-L6-v2) against frozen intent operational descriptions in configs/intents.yaml. No learning step.
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


class MajorityClassifier:
    """Baseline A: Predicts dominant intent class prior. No learning step."""

    def __init__(self):
        self.dominant_class_ = "other_unclear"
        self.classes_ = np.array([self.dominant_class_])

    def fit(self, X, y):
        if y is not None and len(y) > 0:
            counts = pd.Series(y).value_counts()
            self.dominant_class_ = counts.index[0]
            self.classes_ = np.array(sorted(pd.Series(y).unique()))
        return self

    def predict(self, X):
        return np.array([self.dominant_class_] * len(X))

    def predict_proba(self, X):
        n_samples = len(X)
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        dom_idx = np.where(self.classes_ == self.dominant_class_)[0][0]
        probas[:, dom_idx] = 1.0
        return probas


class TfidfLogisticClassifier:
    """Baseline B: TF-IDF + Logistic Regression."""

    def __init__(self, c_param=1.0, ngram_range=(1, 2), min_df=2, random_state=42):
        self.c_param = c_param
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True
        )
        self.clf = LogisticRegression(
            C=self.c_param,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state
        )
        self.classes_ = None

    def fit(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec)

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict_proba(X_vec)

    def save(self, model_path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "vectorizer": self.vectorizer,
            "clf": self.clf,
            "classes_": self.classes_,
            "c_param": self.c_param,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "random_state": self.random_state
        }
        joblib.dump(artifact, model_path)

    @classmethod
    def load(cls, model_path):
        artifact = joblib.load(model_path)
        instance = cls(
            c_param=artifact["c_param"],
            ngram_range=artifact["ngram_range"],
            min_df=artifact["min_df"],
            random_state=artifact["random_state"]
        )
        instance.vectorizer = artifact["vectorizer"]
        instance.clf = artifact["clf"]
        instance.classes_ = artifact["classes_"]
        return instance


class ZeroShotEmbeddingSimilarityBaseline:
    """
    Baseline C: Zero-Shot Embedding Similarity Baseline.
    Cosine similarity between query embedding (all-MiniLM-L6-v2) and frozen intent descriptions in configs/intents.yaml.
    No learning step.
    """

    def __init__(self, config_path="configs/intents.yaml", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.config_path = Path(config_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.intent_ids = []
        self.intent_descriptions = []
        self.intent_embeddings = None
        self._load_intent_descriptions()

    def _load_intent_descriptions(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Intents config missing at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        intents_list = cfg.get("intents", [])
        if isinstance(intents_list, list):
            for item in sorted(intents_list, key=lambda x: x.get("intent_id", "")):
                iid = item.get("intent_id")
                desc = f"{item.get('name', iid)}. {item.get('description', '')}"
                self.intent_ids.append(iid)
                self.intent_descriptions.append(desc)
        elif isinstance(intents_list, dict):
            for iid, details in sorted(intents_list.items()):
                desc = f"{details.get('name', iid)}. {details.get('description', '')}"
                self.intent_ids.append(iid)
                self.intent_descriptions.append(desc)
        
        self.classes_ = np.array(self.intent_ids)
        # Compute frozen intent embeddings
        self.intent_embeddings = self.model.encode(self.intent_descriptions, normalize_embeddings=True)

    def predict(self, X):
        if len(X) == 0:
            return np.array([])
        X_embeddings = self.model.encode(list(X), normalize_embeddings=True)
        sims = np.dot(X_embeddings, self.intent_embeddings.T)  # (n_samples, n_intents)
        best_indices = np.argmax(sims, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X):
        if len(X) == 0:
            return np.zeros((0, len(self.classes_)))
        X_embeddings = self.model.encode(list(X), normalize_embeddings=True)
        sims = np.dot(X_embeddings, self.intent_embeddings.T)  # (n_samples, n_intents)
        # Softmax scaling over cosine similarities as confidence scores
        exp_sims = np.exp(sims * 10.0)  # Temperature scaling 10.0
        probas = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
        return probas


class IntentClassifier:
    """Unified Intent Classifier wrapper providing access to all 3 baselines."""

    def __init__(self, mode="tfidf_logreg", c_param=1.0, config_path="configs/intents.yaml"):
        self.mode = mode
        if mode == "majority":
            self.model = MajorityClassifier()
        elif mode == "tfidf_logreg":
            self.model = TfidfLogisticClassifier(c_param=c_param)
        elif mode == "zero_shot_minilm":
            self.model = ZeroShotEmbeddingSimilarityBaseline(config_path=config_path)
        else:
            raise ValueError(f"Unknown classifier mode: {mode}")

    def fit(self, X, y):
        if self.mode != "zero_shot_minilm":
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
