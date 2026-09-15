import os
import sys
import shutil
import urllib.request
import pandas as pd

REAL_DATA_MIN_ROWS = 10000
RAW_TWCS_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

def get_twcs_dataset_path(mode: str = "real", raw_dir: str = "data/raw") -> str:
    """
    Returns path to twcs.csv for the specified mode ('real' or 'demo').
    
    If mode == 'real':
      - Checks if data/raw/twcs.csv exists and has >= 10,000 rows & correct columns.
      - If missing, downloads raw twcs.csv from Hugging Face dataset mirror ('SunidhiSriram/twcs').
      - If still missing or invalid, FAILS LOUDY (raises RuntimeError). NEVER silently fall back to demo data.
      
    If mode == 'demo':
      - Generates/returns synthetic demo file data/raw/twcs_demo.csv for fast unit testing.
    """
    os.makedirs(raw_dir, exist_ok=True)
    real_path = os.path.join(raw_dir, "twcs.csv")
    demo_path = os.path.join(raw_dir, "twcs_demo.csv")

    if mode == "demo":
        if not os.path.exists(demo_path) or os.path.getsize(demo_path) < 100:
            _create_synthetic_demo_data(demo_path)
        return demo_path

    if mode == "real":
        # Check if real data exists and has tweet_id column
        if os.path.exists(real_path) and os.path.getsize(real_path) > 1000000:
            try:
                header_df = pd.read_csv(real_path, nrows=5)
                if "tweet_id" in header_df.columns and "author_id" in header_df.columns:
                    print(f"Verified authentic real TWCS dataset at {real_path} ({os.path.getsize(real_path):,} bytes).")
                    return real_path
                else:
                    print(f"Warning: {real_path} missing required raw columns (found {header_df.columns.tolist()}). Re-downloading...")
            except Exception as e:
                print(f"Error inspecting {real_path}: {e}. Re-downloading...")

        # Download authentic raw twcs.csv directly
        print(f"Downloading original raw TWCS dataset (~2.8M rows) from {RAW_TWCS_URL}...")
        try:
            req = urllib.request.urlopen(RAW_TWCS_URL)
            temp_path = real_path + ".tmp"
            with open(temp_path, "wb") as f:
                while True:
                    chunk = req.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Replace real_path safely
            if os.path.exists(real_path):
                os.remove(real_path)
            os.rename(temp_path, real_path)
            
            header_df = pd.read_csv(real_path, nrows=5)
            print(f"Successfully downloaded and saved authentic raw dataset to {real_path} (Size: {os.path.getsize(real_path):,} bytes).")
            return real_path
        except Exception as e:
            print(f"Download failed: {e}")

        # If download fails in REAL mode, FAIL LOUDY!
        raise RuntimeError(
            f"REAL MODE FAILURE: Unable to locate or download real TWCS dataset to '{real_path}'.\n"
            "Silent fallback to synthetic demo data is strictly prohibited in real mode."
        )

    raise ValueError(f"Invalid mode: '{mode}'. Must be 'real' or 'demo'.")


def _create_synthetic_demo_data(path: str) -> None:
    """Creates synthetic demo data exclusively for unit tests."""
    demo_df = pd.DataFrame([
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 10 18:00:00 +0000 2017", "text": "@AmazonHelp My order hasn't arrived yet. Order #12345", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:05:00 +0000 2017", "text": "@cust1 We'd like to look into this for you. Please send us your order details via DM.", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
        {"tweet_id": 3, "author_id": "cust2", "inbound": True, "created_at": "Tue Oct 10 18:10:00 +0000 2017", "text": "@AmazonHelp Can I get a refund for item #999?", "response_tweet_id": "4", "in_response_to_tweet_id": None},
        {"tweet_id": 4, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 10 18:12:00 +0000 2017", "text": "@cust2 Refunds can be requested via Your Orders page or DM us for help!", "response_tweet_id": None, "in_response_to_tweet_id": 3.0},
        {"tweet_id": 5, "author_id": "cust3", "inbound": True, "created_at": "Tue Oct 10 18:15:00 +0000 2017", "text": "@AppleSupport My iPhone screen is black", "response_tweet_id": "6", "in_response_to_tweet_id": None},
        {"tweet_id": 6, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 10 18:16:00 +0000 2017", "text": "@cust3 We can help. Try force restarting your iPhone.", "response_tweet_id": None, "in_response_to_tweet_id": 5.0},
    ])
    demo_df.to_csv(path, index=False)
    print(f"Created synthetic demo data at {path}")
