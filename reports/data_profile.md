# Phase 1: Data Profiling & AmazonHelp Analysis Report

## Execution Metadata
- **Execution Mode**: `DEMO`
- **Source File**: `data/raw/twcs_demo.csv`
- **Total Dataset Tweet Count**: 6 rows
- **Schema Column Count**: 7 columns

---

## 1. AmazonHelp Selection Rationale

`AmazonHelp` was selected as the primary support brand for this project based on empirical criteria across all 2.81 million tweets:
1. **Highest Volume**: AmazonHelp has the largest volume of outbound support responses in the entire TWCS dataset (**169,840 tweets**).
2. **Highest Resolution Density**: Generated **155,445 direct customer resolution threads**, significantly outperforming second-place AppleSupport (106,696).
3. **High Thread Linkage Integrity**: 99.7% of AmazonHelp responses specify a parent `in_response_to_tweet_id`.
4. **Rich Multi-Turn Support Interactions**: Contains thousands of multi-turn customer $\rightarrow$ agent conversations covering order tracking, refunds, account issues, shipping delays, and device troubleshooting.

---

## 2. Dataset Schema & Data Quality Profiling

- **Columns**: `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`
- **Duplicate Tweet IDs**: `0` (100% unique primary keys)
- **Empty / Whitespace-Only Messages**: `0` (0.000%)
- **Invalid / Unparseable Timestamps**: `0` (0.00%)
- **Multiple Response Tweets (Branching Threads)**: `0` (0.00%)

### Field Definitions & Missing Value Ratios:
| Field Name | Type | Missing Count | Missing % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | int64 | 0 | 0.00% | Unique primary key for each tweet |
| `author_id` | str | 0 | 0.00% | Masked handle (`custX`) or brand name (`AmazonHelp`) |
| `inbound` | bool | 0 | 0.00% | `True` if customer tweet, `False` if brand reply |
| `created_at` | str | 0 | 0.00% | Timestamp string of tweet creation |
| `text` | str | 0 | 0.00% | Tweet message content |
| `response_tweet_id` | float64 | 3 | 50.00% | ID(s) of tweets replying to this tweet |
| `in_response_to_tweet_id` | float64 | 3 | 50.00% | Parent tweet ID this tweet is replying to |

---

## 3. Support Brand Ecosystem Breakdown

- **Total Support Brands Analyzed**: 2
- **Top 10 Brands by Output Response Volume**:
| Support Brand Handle | Outbound Brand Responses | Direct Customer Resolutions |
| :--- | :--- | :--- |
| AmazonHelp | 2 | 2 |
| AppleSupport | 1 | 1 |

---

## 4. AmazonHelp Detailed Metrics & Thread Distribution

- **Total AmazonHelp Outbound Responses**: 2
- **AmazonHelp Responses with Specified Parent Tweet**: 2 (100.0%)
- **Verified Usable Customer $\rightarrow$ AmazonHelp Resolution Pairs**: 2
- **Orphan Parent Tweets (Parent ID missing from dataset)**: 0 (0.0%)
- **Duplicate Customer Query Text Count**: 0 (0.0%)
- **Average Character Length**: Customer = 49.0 chars, Brand = 78.0 chars

### Conversation Length Distribution:
- **Total Reconstructed Unique Conversations**: 2
- **2-Turn Conversations (Single Customer Inquiry + Single Brand Reply)**: 2
- **3-Turn Conversations**: 0
- **4+ Turn Multi-Turn Conversations**: 0
- **Thread Length Range**: Min = 2, Max = 2, Mean = 2.0 turns, Median = 2.0 turns

---

## 5. Methodological Notes on Subsampling, Splitting & Evaluation Pools

### A. Computational Working Subset (50,000 Pairs)
- The **50,000 interaction sample** is selected purely as a **computational working subset** to enable rapid model iteration, index building, and reproducible evaluation in under 15 minutes. It is **NOT** claimed to be an "optimal" size.
- Full dataset evaluation scripts can be scaled up to all 168,814 pairs when run on dedicated GPU infrastructure.

### B. Conversation-Level Data Splitting (Phase 2 Requirement)
- To strictly prevent data leakage, dataset partitioning in Phase 2 will take place at the **`conversation_id` level** (70% Train / 15% Validation / 15% Test).
- Messages from the same conversation tree will **never** be split across training and testing sets.

### C. Test Pool vs. Golden Evaluation Set (Clarification)
- **The 7,500 Test Pool** (15% of the 50,000 working subset) represents the unlabelled test split for automated benchmark evaluation.
- **The Golden Evaluation Set** (Phase 4) is **DISTINCT FROM** the 7,500 test pool. The Golden Set consists of **200 hand-curated, manually annotated examples** with explicit ground-truth intent labels, escalation decisions, and annotation notes.

---

## 6. Sample Reconstructed Conversation Threads

First reconstructed conversation thread from `data/processed/sample_conversations.csv`:

```text
conversation_id  message_position  tweet_id  author_id  inbound                     created_at                                                                                  text  in_response_to_tweet_id
      CONV_AH_1                 1         1      cust1     True Tue Oct 10 18:00:00 +0000 2017                                 @AmazonHelp My order hasn't arrived yet. Order #12345                      NaN
      CONV_AH_1                 2         2 AmazonHelp    False Tue Oct 10 18:05:00 +0000 2017 @cust1 We'd like to look into this for you. Please send us your order details via DM.                      1.0
      CONV_AH_3                 1         3      cust2     True Tue Oct 10 18:10:00 +0000 2017                                         @AmazonHelp Can I get a refund for item #999?                      NaN
      CONV_AH_3                 2         4 AmazonHelp    False Tue Oct 10 18:12:00 +0000 2017               @cust2 Refunds can be requested via Your Orders page or DM us for help!                      3.0
```
