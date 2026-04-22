from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TRAINING_DIR = PROJECT_DIR / "training"
NOTEBOOKS_DIR = TRAINING_DIR / "notebooks"
SRC_DIR = TRAINING_DIR / "src"
SCRIPTS_DIR = TRAINING_DIR / "scripts"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"

TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
HIERARCHY_DIR = ARTIFACTS_DIR / "hierarchy"
LABEL_MAPS_DIR = ARTIFACTS_DIR / "label_maps"
MODELS_DIR = ARTIFACTS_DIR / "models"

API_DIR = PROJECT_DIR / "api"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"