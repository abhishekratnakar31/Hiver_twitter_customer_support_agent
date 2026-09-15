import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Set
from collections import Counter


def reconstruct_and_filter_interactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconstructs customer -> AmazonHelp interaction records across the entire dataset population
    and applies non-destructive quality filtering.
    """
    target_brand = "AmazonHelp"
    initial_total_rows = len(df)
    
    # Standardize data types
    df_clean = df.copy()
    df_clean["tweet_id"] = df_clean["tweet_id"].astype(np.int64)
    df_clean["datetime"] = pd.to_datetime(df_clean["created_at"], format="mixed", errors="coerce")
    
    # Fast tweet dictionary lookup
    tweet_dict = df_clean.set_index("tweet_id").to_dict("index")
    
    # AmazonHelp outbound tweets
    amazon_outbound = df_clean[(df_clean["author_id"] == target_brand) & (df_clean["inbound"] == False)]
    total_amazon_outbound = len(amazon_outbound)
    
    # Filter responses with parent tweet specified
    amazon_outbound_parent = amazon_outbound[amazon_outbound["in_response_to_tweet_id"].notnull()].copy()
    amazon_outbound_parent["in_response_to_tweet_id"] = amazon_outbound_parent["in_response_to_tweet_id"].astype(np.int64)
    missing_in_resp_id_count = total_amazon_outbound - len(amazon_outbound_parent)
    
    records = []
    orphan_parent_count = 0
    non_inbound_parent_count = 0

    for _, brand_row in amazon_outbound_parent.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        
        if parent_id in tweet_dict:
            parent_tweet = tweet_dict[parent_id]
            if parent_tweet.get("inbound") == True:
                # Traverse to root tweet ID for stable conversation_id
                curr = parent_tweet
                root_id = parent_id
                visited = {parent_id}
                
                while pd.notnull(curr.get("in_response_to_tweet_id")):
                    p_id = int(curr["in_response_to_tweet_id"])
                    if p_id in tweet_dict and p_id not in visited:
                        visited.add(p_id)
                        curr = tweet_dict[p_id]
                        if curr.get("inbound") == True:
                            root_id = p_id
                    else:
                        break
                        
                conv_id = f"CONV_AH_{root_id}"
                inter_id = f"INT_AH_{parent_id}"
                
                cust_text = str(parent_tweet.get("text", "") or "")
                brand_text = str(brand_row.get("text", "") or "")
                
                dt_val = parent_tweet.get("datetime")
                ts_str = dt_val.isoformat() if pd.notnull(dt_val) else str(parent_tweet.get("created_at"))

                records.append({
                    "interaction_id": inter_id,
                    "conversation_id": conv_id,
                    "customer_message": cust_text,
                    "brand_response": brand_text,
                    "timestamp": ts_str,
                    "customer_author_id": parent_tweet.get("author_id"),
                    "customer_tweet_id": parent_id,
                    "brand_tweet_id": brand_row["tweet_id"],
                })
            else:
                non_inbound_parent_count += 1
        else:
            orphan_parent_count += 1

    records_df = pd.DataFrame(records)
    total_reconstructed_raw = len(records_df)

    # QUALITY FILTERING
    # 1. Empty / whitespace-only text filtering
    valid_text_mask = (
        (records_df["customer_message"].str.strip() != "") &
        (records_df["brand_response"].str.strip() != "")
    )
    empty_filtered_count = int((~valid_text_mask).sum())
    clean_df = records_df[valid_text_mask].copy()

    # 2. Duplicate interaction filtering (exact same customer_message & brand_response)
    initial_clean_cnt = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["customer_message", "brand_response"]).copy()
    duplicate_filtered_count = initial_clean_cnt - len(clean_df)

    # 3. Malformed timestamp filtering
    valid_ts_mask = clean_df["timestamp"].notnull() & (clean_df["timestamp"].str.strip() != "")
    clean_df = clean_df[valid_ts_mask].copy()

    filtering_stats = {
        "raw_total_tweets": initial_total_rows,
        "amazonhelp_outbound_responses": total_amazon_outbound,
        "missing_in_response_to_tweet_id_count": missing_in_resp_id_count,
        "orphan_parent_count": orphan_parent_count,
        "non_inbound_parent_count": non_inbound_parent_count,
        "total_reconstructed_pairs_raw": total_reconstructed_raw,
        "empty_whitespace_filtered_count": empty_filtered_count,
        "duplicate_interaction_filtered_count": duplicate_filtered_count,
        "final_clean_interactions": len(clean_df),
        "final_unique_conversations": clean_df["conversation_id"].nunique(),
    }

    return clean_df, filtering_stats


def _compute_turn_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Helper to compute conversation turn length distributions."""
    if df.empty:
        return {"total_convs": 0, "2_turn_pct": 0.0, "3_turn_pct": 0.0, "4_plus_turn_pct": 0.0}
        
    counts = df.groupby("conversation_id").size()
    total = len(counts)
    c2 = (counts == 1).sum() + (counts == 2).sum()  # 1 interaction pair = 2 turns
    c3 = (counts == 3).sum()
    c4 = (counts >= 4).sum()
    
    return {
        "total_convs": total,
        "2_turn_cnt": int(c2),
        "3_turn_cnt": int(c3),
        "4_plus_turn_cnt": int(c4),
        "2_turn_pct": round(float(c2 / total * 100), 2) if total > 0 else 0.0,
        "3_turn_pct": round(float(c3 / total * 100), 2) if total > 0 else 0.0,
        "4_plus_turn_pct": round(float(c4 / total * 100), 2) if total > 0 else 0.0,
    }


def split_conversations_partition_first(
    df_interactions: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_train_interactions: int = 35000,
    target_val_interactions: int = 7500,
    target_test_interactions: int = 7500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes an authoritative conversation-level split on the FULL population first,
    then samples working subsets independently WITHIN each partition.
    """
    np.random.seed(seed)
    
    # Extract ALL unique conversation IDs across full population
    unique_conv_ids = np.array(list(df_interactions["conversation_id"].unique()))
    np.random.shuffle(unique_conv_ids)

    total_convs = len(unique_conv_ids)
    n_train = int(total_convs * train_ratio)
    n_val = int(total_convs * val_ratio)

    full_train_conv_set = set(unique_conv_ids[:n_train])
    full_val_conv_set = set(unique_conv_ids[n_train : n_train + n_val])
    full_test_conv_set = set(unique_conv_ids[n_train + n_val :])

    # Full partition DataFrames
    full_train_df = df_interactions[df_interactions["conversation_id"].isin(full_train_conv_set)].copy()
    full_val_df = df_interactions[df_interactions["conversation_id"].isin(full_val_conv_set)].copy()
    full_test_df = df_interactions[df_interactions["conversation_id"].isin(full_test_conv_set)].copy()

    # Helper function to sample whole conversations within a partition to hit target interaction count
    def sample_subset_within_partition(partition_df: pd.DataFrame, target_interactions: int, partition_seed: int) -> pd.DataFrame:
        if len(partition_df) <= target_interactions or target_interactions <= 0:
            return partition_df.copy()
            
        p_conv_ids = np.array(list(partition_df["conversation_id"].unique()))
        rng = np.random.RandomState(partition_seed)
        rng.shuffle(p_conv_ids)
        
        conv_counts = partition_df.groupby("conversation_id").size().to_dict()
        selected_ids = []
        acc_count = 0
        
        for c_id in p_conv_ids:
            selected_ids.append(c_id)
            acc_count += conv_counts[c_id]
            if acc_count >= target_interactions:
                break
                
        subset_df = partition_df[partition_df["conversation_id"].isin(selected_ids)].copy()
        return subset_df

    # Sample working subsets WITHIN each partition
    train_subset_df = sample_subset_within_partition(full_train_df, target_train_interactions, seed + 1)
    val_subset_df = sample_subset_within_partition(full_val_df, target_val_interactions, seed + 2)
    test_subset_df = sample_subset_within_partition(full_test_df, target_test_interactions, seed + 3)

    train_sub_convs = set(train_subset_df["conversation_id"])
    val_sub_convs = set(val_subset_df["conversation_id"])
    test_sub_convs = set(test_subset_df["conversation_id"])

    # Compute turn length distributions for comparisons
    dist_full_train = _compute_turn_distribution(full_train_df)
    dist_sub_train = _compute_turn_distribution(train_subset_df)
    dist_full_val = _compute_turn_distribution(full_val_df)
    dist_sub_val = _compute_turn_distribution(val_subset_df)
    dist_full_test = _compute_turn_distribution(full_test_df)
    dist_sub_test = _compute_turn_distribution(test_subset_df)

    split_stats = {
        "seed": seed,
        "total_eligible_conversations": total_convs,
        "total_eligible_interactions": len(df_interactions),
        # Full partition stats
        "full_train_conversations": len(full_train_conv_set),
        "full_val_conversations": len(full_val_conv_set),
        "full_test_conversations": len(full_test_conv_set),
        "full_train_interactions": len(full_train_df),
        "full_val_interactions": len(full_val_df),
        "full_test_interactions": len(full_test_df),
        # Working subset stats
        "train_subset_conversations": len(train_sub_convs),
        "val_subset_conversations": len(val_sub_convs),
        "test_subset_conversations": len(test_sub_convs),
        "train_subset_interactions": len(train_subset_df),
        "val_subset_interactions": len(val_subset_df),
        "test_subset_interactions": len(test_subset_df),
        "total_working_interactions": len(train_subset_df) + len(val_subset_df) + len(test_subset_df),
        # Overlap checks
        "full_overlap_train_val": len(full_train_conv_set & full_val_conv_set),
        "full_overlap_train_test": len(full_train_conv_set & full_test_conv_set),
        "full_overlap_val_test": len(full_val_conv_set & full_test_conv_set),
        "subset_overlap_train_val": len(train_sub_convs & val_sub_convs),
        "subset_overlap_train_test": len(train_sub_convs & test_sub_convs),
        "subset_overlap_val_test": len(val_sub_convs & test_sub_convs),
        # Turn length distribution comparison
        "length_distributions": {
            "full_train": dist_full_train,
            "working_train": dist_sub_train,
            "full_val": dist_full_val,
            "working_val": dist_sub_val,
            "full_test": dist_full_test,
            "working_test": dist_sub_test,
        }
    }

    return train_subset_df, val_subset_df, test_subset_df, split_stats
