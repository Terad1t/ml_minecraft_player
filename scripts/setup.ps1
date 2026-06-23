# =============================================================
# setup.ps1 — Setup do projeto minecraft-rl no Windows
# Execute com: .\scripts\setup.ps1
# =============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  minecraft-rl — Setup Windows + DirectML" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verifica pré-requisitos ────────────────────────────────

Write-Host "[1/5] Verificando pre-requisitos..." -ForegroundColor Yellow

# uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  ERRO: 'uv' nao encontrado." -ForegroundColor Red
    Write-Host "  Instale em: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
Write-Host "  OK  uv $(uv --version)" -ForegroundColor Green

# Java (necessário para MineRL)
if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    Write-Host "  ERRO: Java nao encontrado." -ForegroundColor Red
    Write-Host "  MineRL requer Java 8. Instale em: https://adoptium.net/"
    exit 1
}
$javaVersion = java -version 2>&1 | Select-String "version" | Out-String
Write-Host "  OK  Java: $($javaVersion.Trim())" -ForegroundColor Green

# Python 3.10 via uv
Write-Host ""
Write-Host "[2/5] Criando ambiente virtual Python 3.10..." -ForegroundColor Yellow
uv venv --python 3.10
Write-Host "  OK  .venv criado" -ForegroundColor Green

# ── 2. Instala dependências base ──────────────────────────────

Write-Host ""
Write-Host "[3/5] Instalando dependencias (pode demorar)..." -ForegroundColor Yellow

# PyTorch CPU primeiro (DirectML requer versão específica)
Write-Host "  Instalando PyTorch 2.2.2 CPU..." -ForegroundColor Cyan
uv pip install torch==2.2.2 torchvision==0.17.2 `
    --index-url https://download.pytorch.org/whl/cpu

# Instala o resto do pyproject.toml (sem torch, já instalado)
Write-Host "  Instalando demais dependencias..." -ForegroundColor Cyan
uv sync --no-install-package torch --no-install-package torchvision

Write-Host "  OK  Dependencias instaladas" -ForegroundColor Green

# ── 3. Instala torch-directml (AMD GPU) ──────────────────────

Write-Host ""
Write-Host "[4/5] Instalando torch-directml (AMD RX 7600)..." -ForegroundColor Yellow
uv pip install torch-directml
Write-Host "  OK  torch-directml instalado" -ForegroundColor Green

# ── 4. Cria diretórios necessários ───────────────────────────

Write-Host ""
Write-Host "[5/5] Criando diretorios de saida..." -ForegroundColor Yellow

$dirs = @("logs", "logs/tensorboard", "models", "models/checkpoints", "notebooks")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Criado: $dir" -ForegroundColor DarkGray
    }
}
Write-Host "  OK  Diretorios prontos" -ForegroundColor Green

# ── 5. Verifica instalação ────────────────────────────────────

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Verificando instalacao..." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

uv run python -c "
import torch
print(f'  PyTorch: {torch.__version__}')

try:
    import torch_directml
    dml = torch_directml.device()
    t = torch.tensor([1.0]).to(dml) + 1
    print(f'  DirectML: OK — {torch_directml.device_name(0)}')
except Exception as e:
    print(f'  DirectML: FALHOU — {e}')
    print('  Treinamento vai rodar em CPU.')

try:
    import stable_baselines3
    print(f'  SB3: {stable_baselines3.__version__}')
except ImportError:
    print('  SB3: NAO ENCONTRADO')

try:
    import gymnasium
    print(f'  Gymnasium: {gymnasium.__version__}')
except ImportError:
    print('  Gymnasium: NAO ENCONTRADO')
"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  Setup completo!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Proximos passos:" -ForegroundColor Yellow
Write-Host "  1. Rodar agente aleatorio (Fase 1):"
Write-Host "     uv run python -m src.training.random_agent" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Iniciar treino PPO (Fase 2):"
Write-Host "     uv run python -m src.training.train" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Monitorar treinamento:"
Write-Host "     uv run tensorboard --logdir logs/tensorboard" -ForegroundColor Cyan
Write-Host ""