@echo off
title DocuMind Live Deployment
echo ========================================================
echo   DocuMind - Starting Local Server & Public Tunnel...
echo ========================================================
start /b python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true
timeout /t 3 /nobreak >nul
echo.
echo ========================================================
echo   Starting Cloudflare Public Tunnel (Free, No Card)
echo ========================================================
.\cloudflared.exe tunnel --url http://localhost:8501
pause
