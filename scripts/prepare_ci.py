from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "models" / "intent_tfidf_logreg.joblib",
    ROOT / "models" / "rag_faiss.index",
    ROOT / "models" / "rag_metadata.json",
    ROOT / "data" / "processed" / "train.csv",
    ROOT / "data" / "processed" / "validation.csv",
    ROOT / "data" / "processed" / "test.csv",
    ROOT / "data" / "processed" / "test_pool.csv",
    ROOT / "data" / "golden" / "golden_set.csv",
    ROOT / "data" / "golden" / "annotation_template.csv",
    ROOT / "reports" / "golden_set_profile.md",
]

missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(
        "CI artifacts are missing:\n" + "\n".join(missing)
    )

print("All required CI artifacts verified successfully.")
