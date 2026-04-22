from pathlib import Path
import pickle
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 768
PARENT_EMBED_DIM = 64
DROPOUT = 0.1


class HierarchicalEmbedMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        n2,
        n3,
        n4,
        n5,
        n6,
        hidden_dim=HIDDEN_DIM,
        parent_embed_dim=PARENT_EMBED_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.y2_embed = nn.Embedding(n2, parent_embed_dim)
        self.y3_embed = nn.Embedding(n3, parent_embed_dim)
        self.y4_embed = nn.Embedding(n4, parent_embed_dim)
        self.y5_embed = nn.Embedding(n5, parent_embed_dim)

        self.head2 = nn.Linear(hidden_dim, n2)

        self.head3_hidden = nn.Linear(hidden_dim + parent_embed_dim, hidden_dim)
        self.head3 = nn.Linear(hidden_dim, n3)

        self.head4_hidden = nn.Linear(hidden_dim + parent_embed_dim, hidden_dim)
        self.head4 = nn.Linear(hidden_dim, n4)

        self.head5_hidden = nn.Linear(hidden_dim + parent_embed_dim, hidden_dim)
        self.head5 = nn.Linear(hidden_dim, n5)

        self.head6_hidden = nn.Linear(hidden_dim + parent_embed_dim, hidden_dim)
        self.head6 = nn.Linear(hidden_dim, n6)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        return self.input_block(x)


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

    with open(artifacts_dir / "hierarchy" / "hierarchy_embed.pkl", "rb") as f:
        hierarchy = pickle.load(f)

    with open(artifacts_dir / "embedder" / "embed_metadata.pkl", "rb") as f:
        embed_metadata = pickle.load(f)

    embedder_model_name = embed_metadata["model_name"]
    embedder = SentenceTransformer(embedder_model_name, device=DEVICE)

    n2 = len(label_maps["y2"]["classes"])
    n3 = len(label_maps["y3"]["classes"])
    n4 = len(label_maps["y4"]["classes"])
    n5 = len(label_maps["y5"]["classes"])
    n6 = len(label_maps["y6"]["classes"])
    input_dim = int(embed_metadata["embedding_dim"])

    model = HierarchicalEmbedMLP(
        input_dim=input_dim,
        n2=n2,
        n3=n3,
        n4=n4,
        n5=n5,
        n6=n6,
    ).to(DEVICE)

    model_path = artifacts_dir / "models" / "hierarchical_embed_best.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mask23 = torch.tensor(hierarchy["mask23"], dtype=torch.bool).to(DEVICE)
    mask34 = torch.tensor(hierarchy["mask34"], dtype=torch.bool).to(DEVICE)
    mask45 = torch.tensor(hierarchy["mask45"], dtype=torch.bool).to(DEVICE)
    mask56 = torch.tensor(hierarchy["mask56"], dtype=torch.bool).to(DEVICE)

    _artifacts = {
        "device": DEVICE,
        "embedder": embedder,
        "model": model,
        "label_maps": label_maps,
        "mask23": mask23,
        "mask34": mask34,
        "mask45": mask45,
        "mask56": mask56,
        "embed_metadata": embed_metadata,
    }

    return _artifacts