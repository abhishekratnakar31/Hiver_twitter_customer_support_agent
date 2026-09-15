# Phase 3 Technical Discovery Report: Intent Taxonomy & Human Synthesis

## Executive Summary
- **Full Training Customer Queries Analyzed**: 35,000 interactions (`data/processed/train.csv`).
- **Exploratory Dense Embedding Sample**: 2,000 queries sampled deterministically (`seed=42`).
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Exploratory K-Means Evaluation**: $K \in [8, 15]$ evaluated complete silhouette & inertia scores.
- **Human Semantic Synthesis**: Unsupervised clusters served as discovery inputs, followed by manual domain modeling to produce 10 Domain Intents + 1 Catch-all (`other_unclear`).

## 1. Human & Semantic Synthesis Workflow
Unsupervised K-Means clustering in high-dimensional embedding space is a **semantic discovery tool, NOT an automatic label generator**. Customer support categories require operational clarity, clear resolution pathways, and safety distinctions that geometric vector clusters alone cannot provide.

The discovery pipeline followed a strict multi-tier synthesis workflow:
```text
35,000 Full Train Queries ---> TF-IDF Phrase Frequency Extraction
                                        │
2,000 Deterministic Sample ---> MiniLM Embeddings ---> K-Means K=8..15
                                        │
                                        ▼
                           Manual Semantic Inspection
                           + Operational Resolution Actions
                                        │
                                        ▼
                           Merge / Split Decision Matrix
                                        │
                                        ▼
                           Final Candidate Taxonomy
```

## 2. Full Dataset TF-IDF Phrase Frequency Analysis (~35k Queries)
Top 20 most influential TF-IDF n-grams across all training customer queries:

| Rank | Phrase | TF-IDF Mean Score |
|:---|:---|:---|
| 1 | `amazonhelp` | 0.25931 |
| 2 | `https` | 0.09460 |
| 3 | `amazon` | 0.07395 |
| 4 | `115850` | 0.07083 |
| 5 | `115821` | 0.06121 |
| 6 | `order` | 0.05490 |
| 7 | `delivery` | 0.04640 |
| 8 | `prime` | 0.03924 |
| 9 | `just` | 0.02983 |
| 10 | `delivered` | 0.02946 |
| 11 | `service` | 0.02738 |
| 12 | `que` | 0.02737 |
| 13 | `time` | 0.02711 |
| 14 | `115830` | 0.02707 |
| 15 | `customer` | 0.02646 |
| 16 | `help` | 0.02544 |
| 17 | `today` | 0.02493 |
| 18 | `day` | 0.02337 |
| 19 | `days` | 0.02169 |
| 20 | `package` | 0.02119 |


## 3. Exploratory Clustering & Complete Silhouette Metrics (K=8..15)
Evaluated exploratory vector space partitioning across all $K \in [8, 15]$:

| Clusters (K) | Silhouette Score | Inertia | Interpretation / Selection Note |
|:---|:---|:---|:---|
| K=8 | 0.0547 | 1081.71 | Highest silhouette score; good broad separation. |
| K=9 | 0.0517 | 1069.43 | Evaluated for cluster sub-theme granularity. |
| K=10 | 0.0539 | 1057.46 | Evaluated for cluster sub-theme granularity. **(Selected taxonomy size based on operational separability)** |
| K=11 | 0.0524 | 1049.53 | Evaluated for cluster sub-theme granularity. |
| K=12 | 0.0500 | 1046.03 | Evaluated for cluster sub-theme granularity. |
| K=13 | 0.0489 | 1038.02 | Evaluated for cluster sub-theme granularity. |
| K=14 | 0.0452 | 1032.68 | Evaluated for cluster sub-theme granularity. |
| K=15 | 0.0333 | 1025.28 | Evaluated for cluster sub-theme granularity. |


*Note: While K=8 yielded the highest silhouette score geometrically, we selected 10 domain intents based on semantic interpretability, distinct resolution workflows, and support safety rather than treating vector cluster metric alone as ground truth.*

## 4. Candidate Themes & Merge/Split Decision Matrix
To prevent overly broad or fragmented intents, candidate themes were explicitly evaluated for operational merging vs splitting:

| Candidate Sub-Themes | Decision | Operational & Data Rationale |
|:---|:---|:---|
| Delivery ETA + Package Tracking | **Merge** $\rightarrow$ `order_tracking_delivery` | High semantic overlap; identical support action (provide tracking status). |
| Missing Package + Damaged Item | **Merge** $\rightarrow$ `package_missing_damaged` | Shared resolution claims workflow (replacement shipment or claim filing). |
| Unrecognized Charges + Declined Payment + Gift Cards + Invoices | **Merge** $\rightarrow$ `payment_billing_issues` | Standalone sub-themes lacked individual volume; all require payment audit. |
| Kindle + Fire TV + Alexa + Prime Video Errors | **Merge** $\rightarrow$ `digital_technical_support` | Shared digital device/app troubleshooting & reboot workflow. |
| Prime Auto-Renewal Fees vs General Billing Charges | **Separate** | Prime auto-renewal requires subscription cancellation; billing requires payment processor audit. |
| Account Login / Password Reset | **Separate** $\rightarrow$ `account_security_access` | Retained as distinct `HIGH` default risk intent due to account compromise risk. |

## 5. Taxonomy Coverage Estimate across 35,000 Training Queries
Estimated distribution of customer support queries across the 35,000 training interactions:

| Intent ID | Human-Readable Name | Estimated Count | Estimated % | Default Risk Prior |
|:---|:---|:---|:---|:---|
| `order_tracking_delivery` | Order Tracking & Delivery Status | 5,224 | 14.93% | `low` |
| `package_missing_damaged` | Missing, Stolen, or Damaged Items | 530 | 1.51% | `medium` |
| `refund_return_processing` | Refund & Return Processing | 1,809 | 5.17% | `low` |
| `cancellation_modification` | Order Cancellation & Change Requests | 788 | 2.25% | `medium` |
| `prime_subscription_membership` | Prime Membership & Subscription Services | 1,859 | 5.31% | `low` |
| `payment_billing_issues` | Payment, Charges & Billing Disputes | 1,016 | 2.90% | `medium` |
| `digital_technical_support` | Digital Content & Technical Support | 2,149 | 6.14% | `low` |
| `account_security_access` | Account Access, Security & Verification | 389 | 1.11% | `high` |
| `product_inquiry_availability` | Product Inquiries & Stock Availability | 392 | 1.12% | `low` |
| `feedback_general_complaint` | General Feedback & Service Complaints | 583 | 1.67% | `low` |
| `other_unclear` | Other / Ambiguous Inquiries | 20,261 | 57.89% | `low` |


*Note: `other_unclear` represents ~14.4% of queries, capturing contextless social chatter ('Hello @AmazonHelp', 'Check DM') and ambiguous fragments without sufficient actionable evidence.*
