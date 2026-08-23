@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo         DocRetriever - Git Repository Setup ^& Push
echo ============================================================
echo.

cd /d "%~dp0"

:: Step 1: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not found in PATH.
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

:: Step 2: Initialize git if not already initialized
if not exist ".git" (
    echo [1/4] Initializing new Git repository for DocRetriever...
    git init -b main
) else (
    echo [1/4] Existing Git repository detected.
    git branch -M main
)

:: Step 3: Add files and commit
echo [2/4] Staging files...
git add .

echo [3/4] Creating initial commit...
git commit -m "feat: initial commit for DocRetriever - multi-strategy RAG & evaluation harness"

echo.
echo [4/4] Remote Setup ^& Push Options:
echo.
echo If using GitHub CLI (gh):
echo   gh repo create DocRetriever --public --source=. --remote=origin --push
echo.
echo Or if you created a GitHub repo manually (replace URL with your repo URL):
echo   git remote add origin https://github.com/ishanpanwar051/DocRetriever.git
echo   git push -u origin main
echo.
echo ============================================================
echo.
set /p REPO_URL="Enter your GitHub Repo URL (or press Enter to skip remote push): "

if not "%REPO_URL%"=="" (
    git remote remove origin 2>nul
    git remote add origin %REPO_URL%
    echo Pushing to %REPO_URL%...
    git push -u origin main
    if %errorlevel% equ 0 (
        echo.
        echo [SUCCESS] Project pushed successfully to GitHub!
    ) else (
        echo.
        echo [WARNING] Git push failed. Please check your GitHub credentials or repo URL.
    )
)

echo.
pause
