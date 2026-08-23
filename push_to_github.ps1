# push_to_github.ps1 — PowerShell script to initialize and push DocRetriever to GitHub

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $scriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        DocRetriever - Git Repository Setup & Push          " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/" -ForegroundColor Yellow
    exit 1
}

# Step 2: Initialize Git Repo if not present
if (-not (Test-Path "$scriptDir\.git")) {
    Write-Host "[1/4] Initializing new Git repository..." -ForegroundColor Yellow
    git init -b main
} else {
    Write-Host "[1/4] Git repository already initialized." -ForegroundColor Green
    git branch -M main
}

# Step 3: Stage and Commit
Write-Host "[2/4] Staging files..." -ForegroundColor Yellow
git add .

Write-Host "[3/4] Creating commit..." -ForegroundColor Yellow
git commit -m "feat: initial commit for DocRetriever - multi-strategy RAG & evaluation harness"

Write-Host ""
Write-Host "[4/4] Remote configuration" -ForegroundColor Cyan
Write-Host "To push using GitHub CLI (gh):" -ForegroundColor White
Write-Host "  gh repo create DocRetriever --public --source=. --remote=origin --push" -ForegroundColor Gray
Write-Host ""
Write-Host "To push to an existing GitHub repo:" -ForegroundColor White
Write-Host "  git remote add origin https://github.com/ishanpanwar051/DocRetriever.git" -ForegroundColor Gray
Write-Host "  git push -u origin main" -ForegroundColor Gray
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub Repo URL (or press Enter to skip)"
if (-not [string]::IsNullOrWhiteSpace($repoUrl)) {
    git remote remove origin 2>$null
    git remote add origin $repoUrl
    Write-Host "Pushing to $repoUrl..." -ForegroundColor Yellow
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Project pushed successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Push failed. Please verify repository URL and authentication." -ForegroundColor Red
    }
}
