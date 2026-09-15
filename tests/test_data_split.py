import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from src.data.conversations import reconstruct_and_filter_interactions, split_conversations_partition_first
from scripts.prepare_data import run_pipeline


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {"tweet_id": 101, "author_id": "user1", "inbound": True, "created_at": "Tue Oct 10 18:00:00 +0000 2017", "text": "@AmazonHelp Where is my package?", "response_tweet_id": "102", "in_response_to_tweet_id": None},
        {"tweet_id": 102, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:02:00 +0000 2017", "text": "@user1 Please DM us your tracking number.", "response_tweet_id": None, "in_response_to_tweet_id": 101},
        {"tweet_id": 103, "author_id": "user2", "inbound": True, "created_at": "Tue Oct 10 18:10:00 +0000 2017", "text": "@AmazonHelp Cancel my order please", "response_tweet_id": "104", "in_response_to_tweet_id": None},
        {"tweet_id": 104, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:15:00 +0000 2017", "text": "@user2 You can cancel your order from your account dashboard.", "response_tweet_id": None, "in_response_to_tweet_id": 103},
        {"tweet_id": 105, "author_id": "user3", "inbound": True, "created_at": "Tue Oct 10 18:20:00 +0000 2017", "text": "@AmazonHelp Item damaged on arrival", "response_tweet_id": "106", "in_response_to_tweet_id": None},
        {"tweet_id": 106, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:22:00 +0000 2017", "text": "@user3 Sorry to hear that. Please DM us your order ID for replacement.", "response_tweet_id": None, "in_response_to_tweet_id": 105},
        {"tweet_id": 107, "author_id": "user4", "inbound": True, "created_at": "Tue Oct 10 18:30:00 +0000 2017", "text": "@AmazonHelp Refund policy inquiry", "response_tweet_id": "108", "in_response_to_tweet_id": None},
        {"tweet_id": 108, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:32:00 +0000 2017", "text": "@user4 Refunds are processed within 3-5 business days.", "response_tweet_id": None, "in_response_to_tweet_id": 107},
        {"tweet_id": 109, "author_id": "user5", "inbound": True, "created_at": "Tue Oct 10 18:40:00 +0000 2017", "text": "@AmazonHelp Payment failed error", "response_tweet_id": "110", "in_response_to_tweet_id": None},
        {"tweet_id": 110, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:42:00 +0000 2017", "text": "@user5 Please check your payment method or try another card.", "response_tweet_id": None, "in_response_to_tweet_id": 109},
    ])


def test_reconstruct_and_filter_interactions(sample_raw_df):
    clean_df, filter_stats = reconstruct_and_filter_interactions(sample_raw_df)
    assert not clean_df.empty
    assert len(clean_df) == 5
    assert "interaction_id" in clean_df.columns
    assert "conversation_id" in clean_df.columns
    assert "customer_message" in clean_df.columns
    assert "brand_response" in clean_df.columns
    assert filter_stats["final_clean_interactions"] == 5


def test_whole_thread_exact_count_preservation(sample_raw_df):
    clean_df, _ = reconstruct_and_filter_interactions(sample_raw_df)
    train_df, val_df, test_df, _ = split_conversations_partition_first(clean_df, seed=42)

    full_conv_counts = clean_df.groupby("conversation_id").size().to_dict()

    for split_df in [train_df, val_df, test_df]:
        if not split_df.empty:
            split_conv_counts = split_df.groupby("conversation_id").size().to_dict()
            for c_id, count in split_conv_counts.items():
                assert count == full_conv_counts[c_id], f"Whole thread count mismatch for conversation {c_id}: {count} vs {full_conv_counts[c_id]}"


def test_interaction_id_uniqueness(sample_raw_df):
    clean_df, _ = reconstruct_and_filter_interactions(sample_raw_df)
    train_df, val_df, test_df, _ = split_conversations_partition_first(clean_df, seed=42)

    all_inter_ids = list(train_df["interaction_id"]) + list(val_df["interaction_id"]) + list(test_df["interaction_id"])
    assert len(all_inter_ids) == len(set(all_inter_ids))


def test_required_fields_non_null(sample_raw_df):
    clean_df, _ = reconstruct_and_filter_interactions(sample_raw_df)
    train_df, val_df, test_df, _ = split_conversations_partition_first(clean_df, seed=42)

    for split_name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not df_split.empty:
            assert df_split["interaction_id"].isnull().sum() == 0, f"Null interaction_id in {split_name}"
            assert df_split["conversation_id"].isnull().sum() == 0, f"Null conversation_id in {split_name}"
            assert df_split["customer_message"].isnull().sum() == 0, f"Null customer_message in {split_name}"
            assert df_split["brand_response"].isnull().sum() == 0, f"Null brand_response in {split_name}"
            assert df_split["timestamp"].isnull().sum() == 0, f"Null timestamp in {split_name}"


def test_split_reproducibility(sample_raw_df):
    clean_df, _ = reconstruct_and_filter_interactions(sample_raw_df)
    tr1, val1, te1, _ = split_conversations_partition_first(clean_df, seed=42)
    tr2, val2, te2, _ = split_conversations_partition_first(clean_df, seed=42)

    assert list(tr1["interaction_id"]) == list(tr2["interaction_id"])
    assert list(val1["interaction_id"]) == list(val2["interaction_id"])
    assert list(te1["interaction_id"]) == list(te2["interaction_id"])


def test_run_pipeline_demo_mode():
    try:
        run_pipeline(mode="demo", seed=42)
        assert os.path.exists("data/processed/train.csv")
        assert os.path.exists("data/processed/validation.csv")
        assert os.path.exists("data/processed/test.csv")
        assert os.path.exists("data/processed/full_split_summary.json")
        assert os.path.exists("reports/data_split.md")
    finally:
        # Restore real mode data split to prevent test pollution
        run_pipeline(mode="real", seed=42)
