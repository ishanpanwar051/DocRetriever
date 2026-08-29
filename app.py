"""
app.py — Top-level entrypoint for DocRetriever on Streamlit Cloud & local runners.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import ui.dashboard
