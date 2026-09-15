@echo off
echo ===================================================
echo     DocuMind -- Git Commit and Push Automation
echo ===================================================
echo.
cd /d "%~dp0"

echo [1/3] Staging all modified and new files...
git add .

echo [2/3] Committing changes...
git commit -m "feat: upgrade to DocuMind Enterprise RAG (Multilingual, Voice-to-Voice, Document Insights, Domain Profiles, Air-Gapped Privacy)"

echo [3/3] Pushing to remote repository...
git push origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying to push to current branch...
    git push
)

echo.
echo ===================================================
echo   Done! All DocuMind updates pushed successfully.
echo ===================================================
pause
