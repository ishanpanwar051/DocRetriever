# DocuMind -- Git Commit & Push Automation
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "    🧠 DocuMind -- Git Commit & Push Automation" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "[1/3] Staging all files..." -ForegroundColor Yellow
git add .

Write-Host "[2/3] Committing changes..." -ForegroundColor Yellow
git commit -m "feat: upgrade to DocuMind Enterprise RAG (Multilingual, Voice-to-Voice, Document Insights, Domain Profiles, Air-Gapped Privacy)"

Write-Host "[3/3] Pushing to remote..." -ForegroundColor Yellow
git push

Write-Host ""
Write-Host "✅ Done! All DocuMind updates pushed successfully." -ForegroundColor Green
