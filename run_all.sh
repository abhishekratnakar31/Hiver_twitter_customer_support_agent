#!/usr/bin/env bash
# ==============================================================================
# Hiver AmazonHelp AI Support Agent - Master Execution Script
# ==============================================================================
set -e

GREEN='\031[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================================${NC}"
echo -e "${BLUE} Hiver AmazonHelp AI Customer Support Agent - Master Pipeline ${NC}"
echo -e "${BLUE}==============================================================${NC}"

# 1. Check or setup virtual environment
if [ ! -d ".venv" ]; then
    echo -e "\n${YELLOW}[1/4] Creating Virtual Environment (.venv)...${NC}"
    python3 -m venv .venv
else
    echo -e "\n${GREEN}[1/4] Virtual Environment (.venv) Detected.${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# 2. Install / Verify Dependencies
echo -e "\n${YELLOW}[2/4] Installing / Verifying Dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}  Dependencies verified.${NC}"

# 3. Run Pytest Suite
echo -e "\n${YELLOW}[3/4] Running Full Automated Pytest Test Suite (68 Unit Tests)...${NC}"
.venv/bin/pytest tests/

# 4. Run Clean Environment Reproducibility Audit
echo -e "\n${YELLOW}[4/4] Running Clean Environment Reproducibility Audit...${NC}"
.venv/bin/python scripts/verify_clean_reproducibility.py

# 5. Launch Demo Server
echo -e "\n${GREEN}==============================================================${NC}"
echo -e "${GREEN} All tests & verification checks PASSED successfully!          ${NC}"
echo -e "${GREEN} Starting Interactive Multi-Turn Simulator Dashboard...       ${NC}"
echo -e "${GREEN}==============================================================${NC}"
.venv/bin/python scripts/demo_server.py
