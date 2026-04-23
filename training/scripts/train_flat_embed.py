from pathlib import Path
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3

HIDDEN_DIM = 768
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(1)


class FlatEmbedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def topk_accuracy(logits, y, k=5):
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    hits = topk.eq(y.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_n = 0
    correct = 0
    top5 = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
            correct += (torch.argmax(logits, dim=1) == y).sum().item()

            k = min(5, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            top5 += topk.eq(y.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "loss": total_loss / total_n,
        "acc_y6": correct / total_n,
        "top5_y6": top5 / total_n,
    }


def main():
    print("entered main", flush=True)

    project_dir = Path(__file__).resolve().parents[2]

    processed_dir = project_dir / "data" / "processed"
    artifacts_dir = project_dir / "training" / "artifacts"
    label_maps_dir = artifacts_dir / "label_maps"
    embedder_dir = artifacts_dir / "embedder"
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train_embed.npy")
    X_valid = np.load(processed_dir / "X_valid_embed.npy")
    X_test = np.load(processed_dir / "X_test_embed.npy")
    print("loaded X arrays", X_train.shape, X_valid.shape, X_test.shape, flush=True)

    y_train_obj = np.load(processed_dir / "y_train_embed.npz")
    y_valid_obj = np.load(processed_dir / "y_valid_embed.npz")
    y_test_obj = np.load(processed_dir / "y_test_embed.npz")

    y_train = y_train_obj["y6"]
    y_valid = y_valid_obj["y6"]
    y_test = y_test_obj["y6"]
    print("loaded y6 arrays", flush=True)

    with open(label_maps_dir / "label_maps_embed.pkl", "rb") as f:
        label_maps = pickle.load(f)

    with open(embedder_dir / "embed_metadata.pkl", "rb") as f:
        embed_metadata = pickle.load(f)

    input_dim = int(X_train.shape[1])
    n_classes = len(label_maps["y6"]["classes"])

    train_ds = FlatEmbedDataset(X_train, y_train)
    valid_ds = FlatEmbedDataset(X_valid, y_valid)
    test_ds = FlatEmbedDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FlatEmbedMLP(
        input_dim=input_dim,
        n_classes=n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    print("starting training loop", flush=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"starting epoch {epoch}", flush=True)

        model.train()
        running_loss = 0.0
        total_n = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_n = x.size(0)
            running_loss += loss.item() * batch_n
            total_n += batch_n

            if batch_idx % 50 == 0:
                print(f"epoch {epoch} batch {batch_idx} loss {loss.item():.4f}", flush=True)

        train_loss = running_loss / total_n
        valid_metrics = evaluate(model, valid_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_acc_y6": valid_metrics["acc_y6"],
            "valid_top5_y6": valid_metrics["top5_y6"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc_y6={valid_metrics['acc_y6']:.4f} | "
            f"valid_top5_y6={valid_metrics['top5_y6']:.4f}",
            flush=True,
        )

        if valid_metrics["acc_y6"] > best_valid_acc:
            best_valid_acc = valid_metrics["acc_y6"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), models_dir / "flat_embed_best.pt")
            print("saved new best model", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"no improvement for {epochs_without_improvement} epoch(s)", flush=True)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement",
                flush=True,
            )
            break

    print(f"best epoch: {best_epoch}", flush=True)
    print(f"best valid_acc_y6: {best_valid_acc:.4f}", flush=True)

    model.load_state_dict(torch.load(models_dir / "flat_embed_best.pt", map_location=DEVICE))

    print("evaluating test set", flush=True)
    test_metrics = evaluate(model, test_loader, criterion)

    with open(models_dir / "flat_embed_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(models_dir / "flat_embed_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "device": DEVICE,
        "embedder_model_name": embed_metadata["model_name"],
        "embedding_dim": embed_metadata["embedding_dim"],
    }

    with open(models_dir / "flat_embed_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("done", flush=True)
    print("test metrics:", flush=True)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    print("script started", flush=True)
    main()