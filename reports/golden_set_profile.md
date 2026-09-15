# Phase 4 Technical Profile Report: Golden Evaluation Set (200 Examples)

## Executive Summary
- **Golden Evaluation Set Size**: 200 items (`data/golden/golden_set.csv`).
- **Carved-Out Test Pool Size**: 7,300 items (`data/processed/test_pool.csv`).
- **Exact Partition Reconciliation**: 200 Golden + 7,300 Test Pool = 7,500 Original Test interactions (0 missing, 0 extra).
- **Annotation Version**: `phase4_v1`.
- **Conversation Uniqueness**: `nunique(conversation_id) == 200`.
- **Single-Turn vs Multi-Turn**: 200 Single-Turn (100.0%), 0 Multi-Turn (0.0%).
- **Human Escalation Triage Ratio**: 108 Escalate (54.0%), 92 Auto-Handle (46.0%).

## 1. Intent Ground Truth Distribution (200 Golden Examples)
| Intent ID | Human-Readable Name | Count | Percentage | Risk Prior |
|:---|:---|:---|:---|:---|
| `other_unclear` | Other Unclear | 88 | 44.0% | `low` |
| `order_tracking_delivery` | Order Tracking Delivery | 45 | 22.5% | `low` |
| `digital_technical_support` | Digital Technical Support | 21 | 10.5% | `low` |
| `payment_billing_issues` | Payment Billing Issues | 14 | 7.0% | `medium` |
| `product_inquiry_availability` | Product Inquiry Availability | 13 | 6.5% | `low` |
| `cancellation_modification` | Cancellation Modification | 9 | 4.5% | `medium` |
| `refund_return_processing` | Refund Return Processing | 3 | 1.5% | `low` |
| `feedback_general_complaint` | Feedback General Complaint | 3 | 1.5% | `low` |
| `package_missing_damaged` | Package Missing Damaged | 2 | 1.0% | `medium` |
| `account_security_access` | Account Security Access | 1 | 0.5% | `high` |
| `prime_subscription_membership` | Prime Subscription Membership | 1 | 0.5% | `low` |


## 2. Escalation Triage Reasons Breakdown
| Escalation Reason | Count | Percentage | Requires Human Escalation |
|:---|:---|:---|:---|
| `none` | 92 | 46.0% | `False` |
| `ambiguous_request` | 88 | 44.0% | `True` |
| `sensitive_account_issue` | 14 | 7.0% | `True` |
| `complex_unresolved_issue` | 3 | 1.5% | `True` |
| `potential_policy_exception` | 2 | 1.0% | `True` |
| `high_risk` | 1 | 0.5% | `True` |


## 3. Sampling Limitations & Evaluation Provenance
To maintain complete evaluation transparency, the following sampling methodology constraints are explicitly documented:

1. **Exact Partition Reconciliation**: `test.csv` (7,500 interactions) is partitioned into `golden_set.csv` (200 interactions) and `test_pool.csv` (7,300 interactions). Asserted zero missing interactions, zero extra interactions, and zero conversation splitting across splits.
2. **Intent Class Imbalance & Evaluation Metrics**: `other_unclear` represents the largest single class (44.0%). A naive baseline predicting `other_unclear` for all inputs would achieve 44.0% accuracy despite zero utility. Therefore, Phase 5 evaluation MUST report **Macro-F1**, per-intent Precision/Recall/F1, and confusion matrices rather than relying on accuracy alone.
3. **Statistical Reliability for Rare Intents**: Intents with low sample counts (e.g. `account_security_access`: 1, `prime_subscription_membership`: 1, `package_missing_damaged`: 2, `refund_return_processing`: 3, `feedback_general_complaint`: 3) provide qualitative benchmark examples but cannot be interpreted as statistically reliable for per-class performance conclusions.
4. **Interpretation of Escalation Rate**: The human escalation rate in the golden set (54.0%) reflects human-reviewed triage criteria applied to this sampled benchmark, NOT an estimate of Amazon's global real-world escalation rate.
5. **Empirical Historical Responses**: `historical_reference_reply` contains unmodified historical AmazonHelp Twitter agent responses, serving as empirical reference points rather than assumed perfect gold standard answers.
