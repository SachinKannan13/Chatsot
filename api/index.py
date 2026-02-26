import sys
import os
import traceback

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    from main import app  # noqa: F401 — Vercel picks up `app` as the ASGI handler

except Exception as e:
    print("FULL ERROR:")
    print(traceback.format_exc())
    raise e
