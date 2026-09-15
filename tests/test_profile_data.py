import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from src.data.download import get_twcs_dataset_path
from scripts.profile_data import (
    inspect_schema,
    profile_brands,
    analyze_amazonhelp_threads,
    reconstruct_conversations_sample,
    run_profiling,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"tweet_id": 101, "author_id": "user1", "inbound": True, "created_at": "Tue Oct 10 18:00:00 +0000 2017", "text": "@AmazonHelp Where is my package?", "response_tweet_id": "102", "in_response_to_tweet_id": None},
        {"tweet_id": 102, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:02:00 +0000 2017", "text": "@user1 Please DM us your tracking number.", "response_tweet_id": None, "in_response_to_tweet_id": 101},
        {"tweet_id": 103, "author_id": "user2", "inbound": True, "created_at": "Tue Oct 10 18:10:00 +0000 2017", "text": "@AmazonHelp Cancel my order please", "response_tweet_id": "104", "in_response_to_tweet_id": None},
        {"tweet_id": 104, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:15:00 +0000 2017", "text": "@user2 You can cancel your order from your account dashboard.", "response_tweet_id": None, "in_response_to_tweet_id": 103},
        {"tweet_id": 105, "author_id": "user3", "inbound": True, "created_at": "Tue Oct 10 18:20:00 +0000 2017", "text": "@AppleSupport Battery draining fast", "response_tweet_id": "106", "in_response_to_tweet_id": None},
        {"tweet_id": 106, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 10 18:22:00 +0000 2017", "text": "@user3 Check your battery usage in Settings.", "response_tweet_id": None, "in_response_to_tweet_id": 105},
    ])


def test_demo_mode_path_generation(tmp_path):
    demo_path = get_twcs_dataset_path(mode="demo", raw_dir=str(tmp_path))
    assert os.path.exists(demo_path)
    df = pd.read_csv(demo_path)
    assert len(df) == 6
    assert "AmazonHelp" in df["author_id"].values


def test_real_mode_loud_failure_when_missing(tmp_path, monkeypatch):
    """Verifies that real mode FAILS LOUDY if twcs.csv is missing and download fails."""
    import src.data.download
    monkeypatch.setattr(src.data.download, "RAW_TWCS_URL", "https://invalid.url/nonexistent.csv")
    with pytest.raises(RuntimeError, match="REAL MODE FAILURE"):
        get_twcs_dataset_path(mode="real", raw_dir=str(tmp_path))


def test_inspect_schema_integrity_and_data_quality(sample_df):
    schema = inspect_schema(sample_df)
    assert "columns" in schema
    assert "tweet_id" in schema["columns"]
    assert "author_id" in schema["columns"]
    assert "inbound" in schema["columns"]
    assert schema["shape"][0] == 6
    assert schema["duplicate_tweet_ids"] == 0
    assert schema["empty_whitespace_text_count"] == 0
    assert schema["invalid_timestamp_count"] == 0


def test_duplicate_tweet_ids_detection():
    df_dup = pd.DataFrame([
        {"tweet_id": 1, "author_id": "user1", "inbound": True, "created_at": "2017", "text": "Help", "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 1, "author_id": "user1", "inbound": True, "created_at": "2017", "text": "Help", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    schema = inspect_schema(df_dup)
    assert schema["duplicate_tweet_ids"] == 1


def test_missing_empty_text_detection():
    df_empty = pd.DataFrame([
        {"tweet_id": 1, "author_id": "user1", "inbound": True, "created_at": "2017", "text": "   ", "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "user2", "inbound": True, "created_at": "2017", "text": None, "response_tweet_id": None, "in_response_to_tweet_id": None},
        {"tweet_id": 3, "author_id": "user3", "inbound": True, "created_at": "2017", "text": "Valid text", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])
    schema = inspect_schema(df_empty)
    assert schema["empty_whitespace_text_count"] == 2


def test_conversation_id_stability(sample_df):
    _, _, conv_messages_df = analyze_amazonhelp_threads(sample_df)
    assert not conv_messages_df.empty
    conv_ids = conv_messages_df["conversation_id"].unique()
    assert len(conv_ids) == 2
    assert "CONV_AH_101" in conv_ids
    assert "CONV_AH_103" in conv_ids

    # Re-running must produce identical conversation_id
    _, _, conv_messages_df2 = analyze_amazonhelp_threads(sample_df)
    assert list(conv_messages_df["conversation_id"]) == list(conv_messages_df2["conversation_id"])


def test_parent_child_relationships_and_directionality(sample_df):
    _, _, conv_messages_df = analyze_amazonhelp_threads(sample_df)
    
    # Check thread for CONV_AH_101
    thread_101 = conv_messages_df[conv_messages_df["conversation_id"] == "CONV_AH_101"].sort_values("message_position")
    assert len(thread_101) == 2
    
    # Turn 1: Customer inbound
    turn_1 = thread_101.iloc[0]
    assert turn_1["inbound"] == True
    assert turn_1["author_id"] == "user1"
    assert turn_1["tweet_id"] == 101
    
    # Turn 2: AmazonHelp outbound replying to turn 1
    turn_2 = thread_101.iloc[1]
    assert turn_2["inbound"] == False
    assert turn_2["author_id"] == "AmazonHelp"
    assert turn_2["tweet_id"] == 102
    assert int(turn_2["in_response_to_tweet_id"]) == 101


def test_message_ordering_chronological(sample_df):
    _, _, conv_messages_df = analyze_amazonhelp_threads(sample_df)
    for conv_id, group in conv_messages_df.groupby("conversation_id"):
        sorted_group = group.sort_values("message_position")
        timestamps = pd.to_datetime(sorted_group["created_at"])
        assert timestamps.is_monotonic_increasing


def test_run_profiling_demo_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_profiling(mode="demo")
    assert os.path.exists("reports/brand_statistics.csv")
    assert os.path.exists("reports/data_profile.md")
    assert os.path.exists("data/processed/sample_conversations.csv")
