# scripts/setup_ollama.ps1 — Ollama Model Setup for Windows
# 
# Run this script AFTER installing Ollama from https://ollama.com/download
#
# USAGE (in PowerShell, from project root):
#   .\scripts\setup_ollama.ps1
#
# WHY sequential pulls (not parallel):
#   Pulling llama3.2:3b + nomic-embed-text simultaneously would saturate
#   bandwidth AND disk I/O. Sequential = safer, clearer progress.
#   Same principle as our RAM management — never overload resources.

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DocRetriever — Ollama Setup Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Check 1: Ollama installed ──────────────────────────────────────────────────
Write-Host "Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaVersion = & ollama --version 2>&1
    Write-Host "  ✅ Ollama found: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Ollama not found in PATH!" -ForegroundColor Red
    Write-Host "     Download from: https://ollama.com/download/windows" -ForegroundColor White
    Write-Host "     After install, restart PowerShell and re-run this script." -ForegroundColor White
    exit 1
}

# ── Check 2: Ollama server running ────────────────────────────────────────────
Write-Host ""
Write-Host "Checking Ollama server..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "  ✅ Ollama server is running" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Ollama server not responding. Starting it..." -ForegroundColor Yellow
    Write-Host "     If this fails, open a new terminal and run: ollama serve" -ForegroundColor White
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "  Ollama serve started in background." -ForegroundColor Green
}

# ── Pull Model 1: nomic-embed-text (embeddings) ───────────────────────────────
Write-Host ""
Write-Host "Step 1: Pulling nomic-embed-text (embedding model, ~274MB)..." -ForegroundColor Yellow
Write-Host "  WHY nomic-embed-text: 768-dim embeddings, fast on CPU, free" -ForegroundColor Gray
Write-Host ""
& ollama pull nomic-embed-text
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to pull nomic-embed-text" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ nomic-embed-text pulled successfully" -ForegroundColor Green

# ── Quick embedding test ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "Testing nomic-embed-text embedding..." -ForegroundColor Yellow
$embedBody = '{"model": "nomic-embed-text", "input": "test"}'
try {
    $embedResp = Invoke-RestMethod -Uri "http://localhost:11434/api/embed" `
        -Method POST -Body $embedBody -ContentType "application/json" -TimeoutSec 30
    $dim = $embedResp.embeddings[0].Count
    Write-Host "  ✅ Embedding test passed. Dimension: $dim (expected: 768)" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Embedding test failed: $_" -ForegroundColor Yellow
}

# ── Pull Model 2: llama3.2:3b (generation LLM) ────────────────────────────────
Write-Host ""
Write-Host "Step 2: Pulling llama3.2:3b (generation LLM, ~2GB)..." -ForegroundColor Yellow
Write-Host "  WHY llama3.2:3b: smallest capable model for 8GB RAM." -ForegroundColor Gray
Write-Host "  WHY NOT larger: llama3.2:7b would leave <2GB for OS + embeddings." -ForegroundColor Gray
Write-Host "  This download may take 5-15 minutes on first run..." -ForegroundColor Gray
Write-Host ""
& ollama pull llama3.2:3b
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to pull llama3.2:3b" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ llama3.2:3b pulled successfully" -ForegroundColor Green

# ── Verify both models in list ────────────────────────────────────────────────
Write-Host ""
Write-Host "Verifying pulled models..." -ForegroundColor Yellow
$listResp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
$modelNames = $listResp.models | ForEach-Object { $_.name }
Write-Host "  Models available in Ollama:"
$modelNames | ForEach-Object { Write-Host "    - $_" -ForegroundColor Cyan }

# ── RAM Status ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "RAM Status (important for 8GB machines):" -ForegroundColor Yellow
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedGB = [math]::Round($totalGB - $freeGB, 2)
Write-Host "  Total RAM:     $totalGB GB" -ForegroundColor White
Write-Host "  Used RAM:      $usedGB GB" -ForegroundColor White
Write-Host "  Available RAM: $freeGB GB" -ForegroundColor $(if ($freeGB -lt 3) { "Red" } else { "Green" })

if ($freeGB -lt 3) {
    Write-Host "  ⚠️  Low RAM warning! Close other apps before running ingestion." -ForegroundColor Red
} else {
    Write-Host "  ✅ Sufficient RAM for sequential model loading." -ForegroundColor Green
}

# ── Important RAM reminder ────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ✅ Ollama setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  RAM MANAGEMENT REMINDER:" -ForegroundColor Yellow
Write-Host "  - INGESTION: only nomic-embed-text loaded" -ForegroundColor White
Write-Host "  - GENERATION: only llama3.2:3b loaded (keep_alive=0)" -ForegroundColor White
Write-Host "  - RERANKING: only bge-reranker-base loaded" -ForegroundColor White
Write-Host "  - NEVER load 2 models simultaneously!" -ForegroundColor Red
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "  1. docker compose up -d" -ForegroundColor White
Write-Host "  2. python scripts/download_corpus.py" -ForegroundColor White
Write-Host "  3. python scripts/verify_setup.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
