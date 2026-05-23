#!/usr/bin/env bash
set -euo pipefail

# SOC False Positive Reduction - One-Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/jay-chetty-ai/soc-fp-reduction/main/scripts/install.sh | bash

REPO="https://github.com/jay-chetty-ai/soc-fp-reduction.git"
DIR="soc-fp-reduction"
MIN_PYTHON="3.11"

echo ""
echo "  SOC False Positive Reduction - Installer"
echo "  ========================================="
echo ""

# --- Check Python ---
check_python() {
    local cmd
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=$(check_python) || {
    echo "Error: Python ${MIN_PYTHON}+ is required but not found."
    echo "Install Python from https://www.python.org/downloads/"
    exit 1
}
echo "[+] Found Python: $($PYTHON --version)"

# --- Check git ---
if ! command -v git &>/dev/null; then
    echo "Error: git is required but not found."
    exit 1
fi
echo "[+] Found git: $(git --version)"

# --- Clone or update ---
if [ -d "$DIR" ]; then
    echo "[+] Directory $DIR exists, pulling latest..."
    cd "$DIR"
    git pull --ff-only
else
    echo "[+] Cloning repository..."
    git clone "$REPO" "$DIR"
    cd "$DIR"
fi

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[+] Creating virtual environment..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "[+] Virtual environment activated"

# --- Detect CUDA ---
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
    echo "[+] NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
    CUDA_AVAILABLE=true
fi

# --- Install PyTorch ---
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "[+] Installing PyTorch with CUDA support..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "[+] Installing PyTorch (CPU)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

# --- Install dependencies ---
echo "[+] Installing dependencies..."
pip install -q -r requirements.txt

# --- Configure environment ---
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "[!] Created .env file. Add your Anthropic API key:"
    echo "    Edit: $(pwd)/.env"
    echo "    Set:  ANTHROPIC_API_KEY=sk-ant-your-key-here"
    echo ""
fi

# --- Verify ---
echo "[+] Verifying installation..."
python -c "
import lightgbm
import shap
import faiss
import anthropic
import streamlit
print('    All core dependencies OK')
"

if [ "$CUDA_AVAILABLE" = true ]; then
    python -c "
import torch
if torch.cuda.is_available():
    print(f'    CUDA OK: {torch.cuda.get_device_name(0)}')
else:
    print('    CUDA not available (will use CPU for embeddings)')
"
fi

echo ""
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    cd $DIR"
echo "    source .venv/bin/activate"
echo "    pytest tests/ -v --tb=short"
echo "    streamlit run src/ui/dashboard.py"
echo ""
