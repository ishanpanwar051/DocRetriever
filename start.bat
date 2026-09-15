@echo off
cd /d "%~dp0"
echo Starting DocuMind Enterprise RAG Platform...
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe start.py
) else (
    python start.py
)
pause
