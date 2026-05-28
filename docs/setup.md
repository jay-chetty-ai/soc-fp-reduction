# Setup Guide

## Prerequisites

- Python 3.11 or higher
- pip (comes with Python)
- git
- An [Anthropic API key](https://console.anthropic.com/) for Stage 2 LLM triage

**Optional:**
- NVIDIA GPU with CUDA for faster embedding generation (tested with RTX 2070 SUPER, 8GB VRAM)

---

## Manual Setup

### 1. Clone the repository

```bash
git clone <repo-url>
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

To verify CUDA is available:
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

### 5. Download the CICIDS2017 dataset

The pipeline uses the CICIDS2017 ML-ready CSVs from the Canadian Institute for Cybersecurity.

```bash
python scripts/download_data.py
```

This downloads the five weekday CSV files (Monday through Friday) into `data/raw/`. File sizes total roughly 900 MB. If the download helper is not available in your environment, download manually from:

> https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files in `data/raw/`:
```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
└── Friday-WorkingHours-*.pcap_ISCX.csv   (three files)
```

**Note**: The ML CSV release has 78 numeric feature columns plus a `Label` column. It does not have a `Timestamp` column or a `Protocol` column. Timestamps are inferred from filenames during loading.

### 6. Verify installation

```bash
# Check all modules import correctly
python -c "
import lightgbm
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

---

## Three-Step Pipeline Workflow

Once setup is complete, run the pipeline in three steps.

### Step 1: Train Stage 1 model

Trains the LightGBM classifier with Optuna hyperparameter tuning, fits the conformal predictor, and saves both artifacts to `models/`.

```bash
# Fast run (~2 min): skips Optuna tuning, uses default parameters
python scripts/train_stage1.py --skip-tuning

# Full run (~20-40 min): runs 50 Optuna trials
python scripts/train_stage1.py
```

Artifacts saved:
- `models/stage1_model.pkl` -- trained LightGBM model
- `models/conformal.pkl` -- calibrated SplitConformalClassifier
- `models/checksums.json` -- SHA-256 hashes for integrity verification

### Step 2: Build the RAG index

Embeds the training and validation data and builds the FAISS vector index used for historical alert retrieval in Stage 2. Including the validation set (15% of data) ensures all attack families are available as retrieval candidates — with the previous temporal split, Friday-only attack types (DDoS, PortScan, Bot) were absent from the index.

```bash
# Embed a 50K-row sample (faster, good for demos)
python scripts/build_rag_index.py --sample-size 50000

# Embed the full train+val set (~2.45M rows; takes 15-30 min with GPU, longer on CPU)
python scripts/build_rag_index.py
```

Artifacts saved:
- `models/faiss_index.bin` -- FAISS flat inner-product index
- `models/training_df.parquet` -- indexed rows (train + val) aligned to FAISS index positions

### Step 3: Run the pipeline

Processes alerts end-to-end through Stage 1, conformal routing, and Stage 2 LLM adjudication.

```bash
# Process the 10K demo fixture (calls Claude API for uncertain-band alerts)
python scripts/run_pipeline.py \
  --input data/fixtures/fixture_10k.csv \
  --output results/run.csv

# Skip LLM calls (Stage 2 alerts fall back to needs_review)
python scripts/run_pipeline.py \
  --input data/fixtures/fixture_10k.csv \
  --output results/run.csv \
  --no-llm

# Limit to first N alerts (quick smoke test)
python scripts/run_pipeline.py \
  --input data/fixtures/fixture_10k.csv \
  --output results/run.csv \
  --max-alerts 100 \
  --no-llm
```

The script prints a summary on completion:

```
Band distribution:
  auto_fp:   6832 (68.3%)
  uncertain:  1241 (12.4%)
  auto_tp:    1927 (19.3%)

Stage 2 verdicts (uncertain band):
  false_positive:  891 (71.8%)
  true_positive:   287 (23.1%)
  needs_review:     63  (5.1%)

Volume reduction: 78.2% (auto-closed + confirmed FP / total)
Throughput: 4.3 alerts/s
```

### Step 4: Launch the dashboard

```bash
streamlit run src/ui/dashboard.py
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v --tb=short

# Single module
pytest tests/test_epic1_data.py -v --tb=short

# Security controls only
pytest tests/test_security.py -v --tb=short
```

The test suite uses `data/fixtures/fixture_10k.csv` as its primary fixture. Tests that call the Claude API use a mock client. All 157 tests should pass without requiring model artifacts or the full dataset.

---

## GPU Setup Details

The pipeline runs correctly on CPU. GPU accelerates only the embedding step.

**What benefits from GPU:**
- Sentence-transformer embedding generation (10-20x speedup with CUDA)

**What runs on CPU regardless:**
- LightGBM training and inference
- SHAP TreeExplainer
- FAISS similarity search (`faiss-cpu`)
- Claude API calls (remote inference)

**Supported GPUs:**
- Any NVIDIA GPU with CUDA support and 4 GB+ VRAM
- Tested on: NVIDIA RTX 2070 SUPER (8 GB VRAM)

**CUDA notes:**
- PyTorch ships with its own CUDA runtime; you only need the NVIDIA driver installed (not the full CUDA toolkit)
- Check your driver: `nvidia-smi`
- Minimum driver: 525.60+ for CUDA 12.x

The embedding device is controlled by `config.yaml`:
```yaml
rag:
  device: auto      # "auto" = CUDA if available, else CPU
                    # "cpu" to force CPU
                    # "cuda" to require GPU (fails if unavailable)
```

---

## Configuration Reference

All runtime parameters live in `config.yaml`. No operational values are hardcoded in source files.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `data` | `raw_dir` | `data/raw` | Directory containing CICIDS2017 CSV files |
| `data` | `test_day` | `5` | Day number for the legacy temporal hold-out (retained for backward compatibility; primary split is now per-label stratified 70/15/15) |
| `stage1` | `model_artifact_path` | `models/stage1_model.pkl` | Trained LightGBM output path |
| `stage1` | `shap_top_k` | `5` | Number of top SHAP features included in Stage 2 prompts |
| `stage1` | `is_unbalance` | `true` | LightGBM class imbalance handling |
| `tuning` | `n_trials` | `50` | Optuna trial budget |
| `tuning` | `convergence_patience` | `20` | Trials without improvement before early stop |
| `tuning` | `convergence_delta` | `0.001` | Minimum PR-AUC gain to count as improvement |
| `tuning` | `calibration_split` | `0.2` | Legacy: fraction of training data carved out for conformal calibration. Superseded by the per-label stratified val split (15%); retained for backward compatibility |
| `conformal` | `alpha` | `0.05` | Miscoverage rate; gives 95% coverage guarantee |
| `conformal` | `artifact_path` | `models/conformal.pkl` | Conformal predictor output path |
| `stage2` | `model` | `claude-sonnet-4-20250514` | Claude model for Stage 2 adjudication |
| `stage2` | `max_tokens` | `2048` | Max tokens per Stage 2 response |
| `stage2` | `temperature` | `0.1` | Stage 2 temperature |
| `stage2` | `timeout_seconds` | `10` | Anthropic API call timeout |
| `adversarial` | `model` | `claude-sonnet-4-20250514` | Claude model for adversarial pass |
| `adversarial` | `temperature` | `0.3` | Adversarial temperature (higher for diversity) |
| `adversarial` | `confidence_threshold_high` | `0.80` | Stage 2 confidence above which it wins on disagreement |
| `rag` | `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name |
| `rag` | `top_k` | `5` | Number of similar historical alerts to retrieve |
| `rag` | `faiss_index_path` | `models/faiss_index.bin` | FAISS index output path |
| `rag` | `training_df_path` | `models/training_df.parquet` | Training DataFrame for RAG label lookups |
| `rag` | `embedding_batch_size` | `64` | Batch size for SentenceTransformer encode() |
| `rag` | `device` | `auto` | Embedding device: `auto`, `cpu`, or `cuda` |
| `agents` | `max_retries` | `3` | LangGraph retry attempts before fallback to needs_review |
| `agents` | `retry_base_delay_seconds` | `1.0` | Exponential backoff base delay |
| `agents` | `retry_max_delay_seconds` | `30.0` | Exponential backoff cap |
| `tripwire` | `lookback_days` | `7` | IOC retroactive check window in days |
| `a2a` | `mode` | `inprocess` | Agent invocation mode (`inprocess` only; `http` not yet implemented) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'lightgbm'`

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows
```

### `torch.cuda.is_available()` returns `False`

- Check NVIDIA drivers are installed: `nvidia-smi`
- Make sure you installed the CUDA version of PyTorch (see step 3)
- Verify GPU is detected: `lspci | grep -i nvidia`

### FAISS import errors on macOS

```bash
pip install faiss-cpu --no-cache-dir
```

### `ModelIntegrityError` when loading artifacts

The model or conformal artifact does not match the stored SHA-256 hash in `models/checksums.json`. This happens if the file was replaced without going through `save_model()` or `save_conformal()`. Re-run the training scripts to regenerate correct artifacts.

### Anthropic API rate limits

The pipeline retries with exponential backoff (base 1s, max 30s, up to 3 retries). If you are hitting sustained rate limits, reduce the uncertain band by adjusting conformal thresholds in `config.yaml`, or run with `--no-llm` to skip Stage 2 calls entirely.

### `KeyError` or missing artifact errors on `run_pipeline.py`

The script checks for all required artifacts at startup and prints a list of what is missing. Run `train_stage1.py` and `build_rag_index.py` first.
