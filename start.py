"""
start.py — Interactive Single-Command Launcher for DocRetriever

Runs both FastAPI backend and Streamlit UI, checks system health,
and automatically opens the UI in your browser.
"""

import sys
import os
import time
import subprocess
import webbrowser
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable

def print_header():
    print("=" * 65)
    print("        🚀 DocRetriever - All-in-One Launcher")
    print("=" * 65)
    print()

def check_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            print("[INFO] Creating .env from .env.example...")
            env_path.write_bytes(example_path.read_bytes())
            print("[OK] .env created.")

def check_docker():
    print("[1/4] Checking Docker / PostgreSQL...")
    try:
        res = subprocess.run(["docker", "compose", "up", "-d"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print("      ✅ PostgreSQL Docker container is running.")
        else:
            print("      ⚠️ Docker compose returned non-zero. Ensure Docker Desktop is running.")
    except Exception as e:
        print(f"      ⚠️ Docker not detected or not running: {e}")
        print("      (If you are running PostgreSQL locally or in Docker Desktop, please ensure it is started)")

def check_ollama():
    print("[2/4] Checking Ollama...")
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                print("      ✅ Ollama service is active.")
                return
    except Exception:
        pass
    print("      ⚠️ Ollama service not responding at http://localhost:11434")
    print("      Tip: Open terminal and run 'ollama serve' if not running.")

def stream_output(process, prefix):
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[{prefix}] {line.strip()}")

def main():
    print_header()
    check_env()
    check_docker()
    check_ollama()

    print()
    print("[3/4] Starting FastAPI Backend (http://localhost:8000)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    api_cmd = [
        str(VENV_PYTHON), "-m", "uvicorn", "api.main:app",
        "--host", "0.0.0.0", "--port", "8000"
    ]
    api_proc = subprocess.Popen(
        api_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    t_api = threading.Thread(target=stream_output, args=(api_proc, "FastAPI"), daemon=True)
    t_api.start()

    time.sleep(2)

    print("[4/4] Starting Streamlit UI (http://localhost:8501)...")
    streamlit_cmd = [
        str(VENV_PYTHON), "-m", "streamlit", "run", "ui/streamlit_app.py",
        "--server.headless", "false",
        "--server.port", "8501"
    ]
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    t_ui = threading.Thread(target=stream_output, args=(streamlit_proc, "Streamlit"), daemon=True)
    t_ui.start()

    print()
    print("=" * 65)
    print("  ✅ All services started!")
    print("  - Streamlit UI:    http://localhost:8501")
    print("  - FastAPI Backend: http://localhost:8000")
    print("  - Swagger API:     http://localhost:8000/docs")
    print("=" * 65)
    print()
    print("Opening browser in 3 seconds... (Press Ctrl+C to stop all servers)")
    print()

    time.sleep(3)
    webbrowser.open("http://localhost:8501")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping DocRetriever services...")
        api_proc.terminate()
        streamlit_proc.terminate()
        print("Done. Goodbye!")

if __name__ == "__main__":
    main()
