# Twitter Customer Support AI Agent & Evaluation Harness

An enterprise-grade, evaluation-first AI Customer Support Agent and rigorous testing system built for **`AmazonHelp`** using the **Customer Support on Twitter (TWCS)** dataset (~2.81 million tweets).

---

## Key Highlights

- **Target Brand Focus**: Optimized specifically for `AmazonHelp`, empirically proven to be the largest support handle in the TWCS dataset (**169,840 outbound responses**, **155,445 direct customer resolutions**).
- **Leakage-Free Conversation-Level Splitting**: Partitioned the entire population of **82,623 unique conversations** first (70% Train / 15% Val / 15% Test) with `seed=42`, guaranteeing **EXACTLY 0 conversation overlap** between splits.
- **Computational Working Subsets**: Sampled whole conversation threads within each partition to construct working subsets (**35,000 Train**, **7,503 Val**, **7,500 Test** interactions), preserving multi-turn thread length distributions (~77% 2-turn, ~11.5% 3-turn, ~11.5% 4+ turn).
- **100% Reproducible Pipeline**: Full evaluation and data preparation suite runs in **under 15 minutes** with clear command-line interfaces.
- **Strict Mode Isolation**: Supports `--source real` for research execution on 2.81M rows (fails loudly if real data is missing) and `--source demo` for fast offline unit testing.

---

## Repository Structure

```text
hiver-support-agent/
├── README.md                      # Project documentation and quickstart guide
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment variable template for API keys
├── .gitignore                     # Git ignore rules
│
├── configs/
│   ├── config.yaml                # Project configuration file
│   └── intents.yaml               # (Phase 3) Intent taxonomy definitions
│
├── data/
│   ├── README.md                  # Data layout and subsampling notes
│   ├── raw/                       # (Ignored) Raw twcs.csv dataset location
│   └── processed/                 # Sample conversations & split metadata
│       ├── train.csv              # 35,000 clean training interactions
│       ├── validation.csv         # 7,503 clean validation interactions
│       ├── test.csv               # 7,500 unlabelled test pool interactions
│       ├── full_split_summary.json# Detailed partition & subset metadata
│       └── sample_conversations.csv# 25 reconstructed sample conversation threads
│
├── reports/
│   ├── data_profile.md            # Phase 1 empirical data profiling report
│   ├── brand_statistics.csv       # Ranking of top 20 Twitter support handles
│   └── data_split.md              # Phase 2 partition & leakage proof report
│
├── src/
│   ├── __init__.py
│   ├── data/                      # Data downloading, reconstruction & splitting
│   │   ├── __init__.py
│   │   ├── download.py            # Real/demo dataset resolver & downloader
│   │   └── conversations.py       # Thread reconstruction & leakage-safe splitter
│   ├── agent/                     # (Phases 5-11) Agent modules
│   └── evaluation/                # (Phases 12-15) Evaluation harness & LLM judge
│
├── scripts/
│   ├── profile_data.py            # Phase 1 dataset discovery CLI
│   └── prepare_data.py            # Phase 2 interaction pairing & splitting CLI
│
└── tests/
    ├── test_profile_data.py       # Phase 1 unit tests (9 tests)
    └── test_data_split.py         # Phase 2 unit tests (6 tests)
```

---

## Quickstart & Reproducibility (< 15 Minutes)

### 1. Prerequisites & Virtual Environment

Ensure Python 3.9+ is installed. Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/hiver-support-agent.git
cd hiver-support-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the `.env.example` file to create your `.env` configuration:

```bash
cp .env.example .env
```

### 3. Run Automated Unit Tests

Run the complete test suite across data profiling, schema validation, thread reconstruction, and zero-leakage assertions:

```bash
pytest tests/
```

*Expected output*: `15 passed` in under 2 seconds.

---

## Executing the Data Pipeline

### Step 1: Run Dataset Discovery & Profiling (Phase 1)

Runs dataset schema inspection, profiles the top support handles across 2.81M tweets, and generates `reports/data_profile.md`:

```bash
# Research mode on real 2.81M dataset
python scripts/profile_data.py --source real

# Fast demo mode (unit testing)
python scripts/profile_data.py --source demo
```

### Step 2: Run Leakage-Safe Data Splitting (Phase 2)

Reconstructs all customer $\rightarrow$ `AmazonHelp` interaction threads, partitions the full population (70% Train / 15% Val / 15% Test) with `seed=42`, samples working subsets, and exports processed CSVs:

```bash
# Research mode on real dataset
python scripts/prepare_data.py --source real --seed 42

# Fast demo mode
python scripts/prepare_data.py --source demo
```

---

## Summary of Data Split Statistics

### Authoritative Full Population Partitions (82,623 Unique Conversations)
- **Full Train Partition (70%)**: 117,802 interactions (**57,836 conversations**)
- **Full Validation Partition (15%)**: 25,312 interactions (**12,393 conversations**)
- **Full Test Partition (15%)**: 25,700 interactions (**12,394 conversations**)
- **Authoritative Conversation Overlap**: **EXACTLY 0**

### Computational Working Subsets (Processed Files)
- **`data/processed/train.csv`**: **35,000 interactions** (17,097 conversations)
- **`data/processed/validation.csv`**: **7,503 interactions** (3,657 conversations)
- **`data/processed/test.csv`**: **7,500 interactions** (3,653 conversations)
- **Working Subset Conversation Overlap**: **EXACTLY 0**

---

## License & Attribution

- Primary Dataset: Kaggle `thoughtvector/customer-support-on-twitter` (Kaggle License).
- Codebase License: MIT License.
