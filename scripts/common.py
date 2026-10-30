import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for d in (DATA, MODELS, RESULTS):
    d.mkdir(exist_ok=True)
