# Setup Guide

## Prerequisites

- Python 3.11 or higher
- pip (comes with Python)
- git
- An [Anthropic API key](https://console.anthropic.com/) for Stage 2 LLM triage

**Optional:**
- NVIDIA GPU with CUDA for faster embedding generation (tested with RTX 2070 SUPER, 8GB VRAM)

## One-Line Install

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/jay-chetty-ai/soc-fp-reduction/main/scripts/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/jay-chetty-ai/soc-fp-reduction/main/scripts/install.ps1 | iex
```

## Manual Setup

### 1. Clone the repository

```bash
git clone https://github.com/jay-chetty-ai/soc-fp-reduction.git
cd soc-fp-reduction
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

**CPU only (default):**
```bash
pip install -r requirements.txt
```

**With GPU support (NVIDIA CUDA):**
```bash
# Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements.txt
```

To check if CUDA is available after install:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Verify installation

```bash
# Check all modules import correctly
python -c "
import lightgbm
import xgboost
import shap
import mapie
import sentence_transformers
import faiss
import anthropic
import streamlit
print('All dependencies installed successfully.')
"

# Run the test suite
pytest tests/ -v --tb=short
```

## GPU Setup Details

The pipeline is designed to work on CPU. GPU acceleration is used only for generating sentence-transformer embeddings (MiniLM-L6-v2).

**What benefits from GPU:**
- Embedding generation (10-20x speedup with CUDA)

**What runs on CPU regardless:**
- LightGBM / XGBoost training and inference
- SHAP TreeExplainer
- FAISS similarity search (using faiss-cpu)
- Claude API calls (remote inference)

**Supported GPUs:**
- Any NVIDIA GPU with CUDA support and 4GB+ VRAM
- Tested on: NVIDIA RTX 2070 SUPER (8GB VRAM)

**CUDA versions:**
- PyTorch ships with its own CUDA runtime, so you only need the NVIDIA driver installed (not the full CUDA toolkit)
- Check your driver version: `nvidia-smi`
- Minimum driver: 525.60+ for CUDA 12.x support

## Troubleshooting

### `ModuleNotFoundError: No module named 'lightgbm'`
Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows
```

### `torch.cuda.is_available()` returns `False`
- Check that NVIDIA drivers are installed: `nvidia-smi`
- Make sure you installed the CUDA version of PyTorch (see step 3)
- Verify your GPU is supported: `lspci | grep -i nvidia`

### FAISS import errors on macOS
```bash
# If faiss-cpu fails, try:
pip install faiss-cpu --no-cache-dir
```

### Anthropic API rate limits
Stage 2 calls Claude API for each uncertain alert. Default rate limits apply. For batch processing, the pipeline includes automatic retry with backoff. Set your rate tier in `config.yaml` if needed.
