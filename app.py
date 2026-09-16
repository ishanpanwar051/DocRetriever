"""
app.py — Top-level entrypoint for DocRetriever on Streamlit Cloud & local runners.
"""
import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

dashboard_path = ROOT_DIR / "ui" / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")

