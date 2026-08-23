@echo off
cd /d "%~dp0"
title DocRetriever Launcher
echo ============================================================
echo           DocRetriever - Multi-Strategy RAG Launcher
echo ============================================================
echo.

:: 1. Check .env file
if not exist .env (
    echo [INFO] .env file not found. Creating from .env.example...
    copy .env.example .env
    echo [OK] .env created.
)

:: 2. Check and start Docker container
echo [1/3] Checking PostgreSQL Docker container...
docker compose up -d
if %errorlevel% neq 0 (
    echo [WARNING] Docker command failed. Please ensure Docker Desktop is running!
) else (
    echo [OK] PostgreSQL container started/verified.
)
echo.

:: 3. Check virtual environment
echo [2/3] Checking Python Virtual Environment...
if exist venv\Scripts\python.exe (
    echo [OK] Found virtual environment in venv.
) else (
    echo [WARNING] venv not found! Creating venv...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)
echo.

:: 4. Start FastAPI Backend in a new window
echo [3/3] Starting Services...
echo Launching FastAPI backend on http://localhost:8000 ...
start "DocRetriever - FastAPI Backend" cmd /k "cd /d ""%~dp0"" && call venv\Scripts\activate.bat && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: 5. Start Streamlit UI in a new window
echo Launching Streamlit UI on http://localhost:8501 ...
start "DocRetriever - Streamlit UI" cmd /k "cd /d ""%~dp0"" && call venv\Scripts\activate.bat && python -m streamlit run ui\streamlit_app.py"

echo.
echo ============================================================
echo   DocRetriever is launching in separate command windows!
echo   - FastAPI Backend: http://localhost:8000 (Docs: http://localhost:8000/docs)
echo   - Streamlit UI:    http://localhost:8501
echo ============================================================
echo.
pause
