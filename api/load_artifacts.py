from pathlib import Path
import pickle
import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 768
DROPOUT = 0.1


class FlatEmbedMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


_artifacts = None


def _artifacts_root():
    return Path(__file__).resolve().parents[1] / "training" / "artifacts"


def load_artifacts():
    global _artifacts

    if _artifacts is not None:
        return _artifacts

    artifacts_dir = _artifacts_root()

    with open(artifacts_dir / "label_maps" / "label_maps_embed.pkl", "rb") as f:
        label_maps = pickle.load(f)

    with open(artifacts_dir / "embedder" / "embed_metadata.pkl", "rb") as f:
        embed_metadata = pickle.load(f)

    embedder_model_name = embed_metadata["model_name"]
    embedder = SentenceTransformer(embedder_model_name, device=DEVICE)

    n_classes = len(label_maps["y6"]["classes"])
    input_dim = int(embed_metadata["embedding_dim"])

    model = FlatEmbedMLP(
        input_dim=input_dim,
        n_classes=n_classes,
    ).to(DEVICE)

    model_path = artifacts_dir / "models" / "flat_embed_best.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    supplement_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "2022_naics_supplemental.xlsx"
    supplement_df = pd.read_excel(supplement_path)

    supplement_df["naics_code"] = supplement_df["naics_code"].astype(str).str.strip()
    supplement_df["naics_title"] = supplement_df["naics_title"].astype(str).str.strip()

    y6_title_lookup = dict(
        zip(supplement_df["naics_code"], supplement_df["naics_title"])
    )


    _artifacts = {
        "device": DEVICE,
        "embedder": embedder,
        "model": model,
        "label_maps": label_maps,
        "embed_metadata": embed_metadata,
        "y6_title_lookup": y6_title_lookup,
    }

    return _artifacts