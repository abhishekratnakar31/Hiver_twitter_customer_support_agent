# Data Directory Overview

This directory contains the dataset files and processing outputs for the `AmazonHelp` Twitter AI Support Agent.

---

## Directory Structure

- `raw/`: Stores the raw TWCS dataset (`twcs.csv`). This file is ignored by Git due to file size (~516 MB). The download script `src/data/download.py` automatically downloads `twcs.csv` on demand if missing.
- `processed/`: Stores clean, leakage-free interaction splits and conversation samples:
  - `train.csv`: 35,000 clean training interactions (17,097 conversations).
  - `validation.csv`: 7,503 clean validation interactions (3,657 conversations).
  - `test.csv`: 7,500 unlabelled test pool interactions (3,653 conversations).
  - `full_split_summary.json`: JSON metadata recording full population partitions and working subset counts.
  - `sample_conversations.csv`: 25 reconstructed multi-turn conversation threads.
- `golden/`: (Phase 4) Will contain the 200 hand-labelled Golden Evaluation examples.

---

## Data Leakage Safeguards

Dataset splitting is performed strictly at the **`conversation_id` level** using fixed `seed=42`. All interactions belonging to a conversation tree are mapped exclusively to a single partition, guaranteeing zero conversation ID overlap across splits.
