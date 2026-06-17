from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
REPORTS_DIR = ROOT_DIR / "reports"

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for directory in [DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR, GRAPHS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
