import sys
import os

# add project root to path so `main` and `src` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401  — Vercel picks up `app` as the ASGI handler
