#!/usr/bin/env python3
"""
scripts/demo_server.py

Phase 9 API Server & Dashboard Launcher:
Starts Uvicorn server hosting FastAPI app on http://127.0.0.1:8000 and automatically opens the dashboard.
"""

import os
import sys
import uvicorn
import webbrowser
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def open_browser():
    time.sleep(1.5) # Wait for uvicorn to start
    print("\n[Browser] Opening dashboard automatically...")
    webbrowser.open("http://127.0.0.1:8000/dashboard")

def main():
    print("=" * 60)
    print("HIVER PHASE 9: FASTAPI API SERVER & WEB SUPPORT DASHBOARD")
    print("=" * 60)
    print("Dashboard    : http://127.0.0.1:8000/dashboard")
    print("API Endpoint : http://127.0.0.1:8000/api/v1/inquire")
    print("Health Status: http://127.0.0.1:8000/api/v1/health")
    print("Metrics API  : http://127.0.0.1:8000/api/v1/metrics")
    print("Docs (OpenAPI): http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    # Start browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
