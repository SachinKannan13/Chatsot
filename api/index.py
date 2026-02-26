import sys
import os

# Get root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add root to Python path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Now import FastAPI app
from main import app