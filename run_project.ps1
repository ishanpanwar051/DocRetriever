# run_project.ps1 — PowerShell Launcher for DocRetriever

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $scriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         DocRetriever - Multi-Strategy RAG Launcher         " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check .env file
if (-not (Test-Path "$scriptDir\.env")) {
    Write-Host "[INFO] .env file not found. Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item "$scriptDir\.env.example" "$scriptDir\.env"
    Write-Host "[OK] .env created." -ForegroundColor Green
}

# 2. Check Docker
Write-Host "[1/3] Starting PostgreSQL Docker container..." -ForegroundColor Yellow
try {
    docker compose up -d
    Write-Host "[OK] PostgreSQL container running." -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Could not start Docker container. Ensure Docker Desktop is running." -ForegroundColor Red
}

# 3. Check venv
Write-Host "[2/3] Checking Python Virtual Environment..." -ForegroundColor Yellow
$venvPython = "$scriptDir\venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "[OK] Virtual environment found." -ForegroundColor Green
} else {
    Write-Host "[WARNING] venv not found! Creating venv..." -ForegroundColor Yellow
    python -m venv "$scriptDir\venv"
    & "$venvPython" -m pip install --upgrade pip
    & "$venvPython" -m pip install -r "$scriptDir\requirements.txt"
}

# 4. Start FastAPI in new terminal
Write-Host "[3/3] Launching Backend & Frontend..." -ForegroundColor Yellow
Write-Host "  -> Starting FastAPI Backend (http://localhost:8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "Set-Location '$scriptDir'; & '$venvPython' -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

# 5. Start Streamlit in new terminal
Write-Host "  -> Starting Streamlit UI (http://localhost:8501)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "Set-Location '$scriptDir'; & '$venvPython' -m streamlit run ui\streamlit_app.py"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DocRetriever services launched in separate windows!" -ForegroundColor Green
Write-Host "  - FastAPI Backend: http://localhost:8000 (Docs: http://localhost:8000/docs)" -ForegroundColor White
Write-Host "  - Streamlit UI:    http://localhost:8501" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
