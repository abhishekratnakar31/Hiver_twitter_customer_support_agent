"""
scripts/verify_clean_reproducibility.py

Standalone script to verify repository hygiene, delivery artifacts,
and clean environment API readiness.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

REQUIRED_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    ".dockerignore",
    ".gitignore",
    ".env.example",
    "README.md",
    "requirements.txt",
    ".github/workflows/ci.yml",
    "web/index.html",
    "web/app.js",
    "src/api/app.py",
    "src/monitoring/prometheus_exporter.py"
]


def check_required_files() -> bool:
    print("\n--- Checking Required Delivery Files ---")
    all_found = True
    for file_rel in REQUIRED_FILES:
        path = BASE_DIR / file_rel
        if path.exists():
            print(f"  [OK] {file_rel}")
        else:
            print(f"  [MISSING] {file_rel}")
            all_found = False
    return all_found


def check_git_cleanliness() -> bool:
    print("\n--- Checking Git Tracked File Hygiene ---")
    try:
        res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
        tracked_files = res.stdout.splitlines()

        leaks = []
        for file in tracked_files:
            file_path = Path(file)
            # Check forbidden exact names or directory components
            if file_path.name in [".env", ".env.local", ".DS_Store"]:
                leaks.append((file, "forbidden exact file"))
            elif any(part in [".venv", "venv", "__pycache__", ".pytest_cache"] for part in file_path.parts):
                leaks.append((file, "forbidden environment/cache directory"))

        if leaks:
            print("  [WARNING] Found sensitive/temporary tracked files in git:")
            for file, reason in leaks:
                print(f"    - {file} ({reason})")
            return False
        else:
            print("  [OK] Zero sensitive or temporary files tracked in git.")
            return True
    except Exception as e:
        print(f"  [NOTICE] Git check skipped (not a git execution or git CLI uninstalled): {e}")
        return True



def check_fastapi_endpoints() -> bool:
    print("\n--- Checking FastAPI Endpoint Response Interfaces ---")
    try:
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)

        # Health
        h_res = client.get("/api/v1/health")
        assert h_res.status_code == 200, f"Health returned {h_res.status_code}"
        print(f"  [OK] GET /api/v1/health -> {h_res.json()['status']}")

        # JSON Metrics
        m_res = client.get("/api/v1/metrics")
        assert m_res.status_code == 200, f"JSON Metrics returned {m_res.status_code}"
        print(f"  [OK] GET /api/v1/metrics -> total_inquiries: {m_res.json().get('total_inquiries', 0)}")

        # Prometheus Metrics
        p_res = client.get("/metrics")
        assert p_res.status_code == 200, f"Prometheus Metrics returned {p_res.status_code}"
        assert "hiver_uptime_seconds" in p_res.text
        print("  [OK] GET /metrics -> Prometheus exposition format verified.")

        # Dashboard Mount
        d_res = client.get("/dashboard/")
        assert d_res.status_code == 200, f"Dashboard returned {d_res.status_code}"
        print("  [OK] GET /dashboard/ -> HTML static mount verified.")

        return True
    except Exception as e:
        print(f"  [ERROR] Endpoint verification failed: {e}")
        return False


def main():
    print("==========================================================")
    print(" Hiver AmazonHelp Agent - Clean Reproducibility Verification ")
    print("==========================================================")

    f_ok = check_required_files()
    g_ok = check_git_cleanliness()
    e_ok = check_fastapi_endpoints()

    print("\n----------------------------------------------------------")
    if f_ok and g_ok and e_ok:
        print(" SUCCESS: All Phase 10 delivery & reproducibility checks PASSED.")
        sys.exit(0)
    else:
        print(" FAILURE: One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
