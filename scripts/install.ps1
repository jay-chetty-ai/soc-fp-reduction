# SOC False Positive Reduction - One-Line Installer (Windows)
# Usage: irm https://raw.githubusercontent.com/jay-chetty-ai/soc-fp-reduction/main/scripts/install.ps1 | iex

$ErrorActionPreference = "Stop"
$Repo = "https://github.com/jay-chetty-ai/soc-fp-reduction.git"
$Dir = "soc-fp-reduction"
$MinPython = [version]"3.11"

Write-Host ""
Write-Host "  SOC False Positive Reduction - Installer" -ForegroundColor Cyan
Write-Host "  =========================================" -ForegroundColor Cyan
Write-Host ""

# --- Check Python ---
$PythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ([version]$ver -ge $MinPython) {
            $PythonCmd = $cmd
            break
        }
    } catch { }
}

if (-not $PythonCmd) {
    Write-Host "Error: Python $MinPython+ is required but not found." -ForegroundColor Red
    Write-Host "Install Python from https://www.python.org/downloads/"
    exit 1
}
Write-Host "[+] Found Python: $(& $PythonCmd --version)"

# --- Check git ---
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git is required but not found." -ForegroundColor Red
    exit 1
}
Write-Host "[+] Found git: $(git --version)"

# --- Clone or update ---
if (Test-Path $Dir) {
    Write-Host "[+] Directory $Dir exists, pulling latest..."
    Set-Location $Dir
    git pull --ff-only
} else {
    Write-Host "[+] Cloning repository..."
    git clone $Repo $Dir
    Set-Location $Dir
}

# --- Create virtual environment ---
if (-not (Test-Path ".venv")) {
    Write-Host "[+] Creating virtual environment..."
    & $PythonCmd -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
Write-Host "[+] Virtual environment activated"

# --- Detect CUDA ---
$CudaAvailable = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $GpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    Write-Host "[+] NVIDIA GPU detected: $GpuName"
    $CudaAvailable = $true
}

# --- Install PyTorch ---
if ($CudaAvailable) {
    Write-Host "[+] Installing PyTorch with CUDA support..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "[+] Installing PyTorch (CPU)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
}

# --- Install dependencies ---
Write-Host "[+] Installing dependencies..."
pip install -q -r requirements.txt

# --- Configure environment ---
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host ""
    Write-Host "[!] Created .env file. Add your Anthropic API key:" -ForegroundColor Yellow
    Write-Host "    Edit: $(Get-Location)\.env"
    Write-Host "    Set:  ANTHROPIC_API_KEY=sk-ant-your-key-here"
    Write-Host ""
}

# --- Verify ---
Write-Host "[+] Verifying installation..."
python -c @"
import lightgbm
import shap
import faiss
import anthropic
import streamlit
print('    All core dependencies OK')
"@

if ($CudaAvailable) {
    python -c @"
import torch
if torch.cuda.is_available():
    print(f'    CUDA OK: {torch.cuda.get_device_name(0)}')
else:
    print('    CUDA not available (will use CPU for embeddings)')
"@
}

Write-Host ""
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    cd $Dir"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    pytest tests/ -v --tb=short"
Write-Host "    streamlit run src\ui\dashboard.py"
Write-Host ""
