@echo off
cd /d "%~dp0"
echo Starting DocRetriever...
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe start.py
) else (
    python start.py
)
pause
