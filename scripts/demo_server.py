
#!/usr/bin/env python3
"""
scripts/demo_server.py

Phase 7 Production API Server & Dashboard Launcher:
Starts Uvicorn server hosting FastAPI app on http://127.0.0.1:8000 and mounts static web/ dashboard.
"""

import os
import sys
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.api.app import app

# Mount static web directory for static dashboard hosting
web_dir = BASE_DIR / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


def main():
    print("=" * 60)
    print("HIVER PHASE 7: FASTAPI API SERVER & WEB SUPPORT DASHBOARD")
    print("=" * 60)
    print("Server URL   : http://127.0.0.1:8000")
    print("API Endpoint : http://127.0.0.1:8000/api/v1/inquire")
    print("Health Status: http://127.0.0.1:8000/api/v1/health")
    print("Metrics API  : http://127.0.0.1:8000/api/v1/metrics")
    print("Docs (OpenAPI): http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
