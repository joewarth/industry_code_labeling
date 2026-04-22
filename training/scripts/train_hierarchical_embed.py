from pathlib import Path
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3

HIDDEN_DIM = 768
PARENT_EMBED_DIM = 64
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(1)


class HierEmbedDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y2 = torch.tensor(y_dict["y2"], dtype=torch.long)
        self.y3 = torch.tensor(y_dict["y3"], dtype=torch.long)
        self.y4 = torch.tensor(y_dict["y4"], dtype=torch.long)
        self.y5 = torch.tensor(y_dict["y5"], dtype=torch.long)
        self.y6 = torch.tensor(y_dict["y6"], dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y2[idx],
            self.y3[idx],
            self.y4[idx],
            self.y5[idx],
            self.y6[idx],
        )


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

    def forward_teacher_forced(self, x, y2, y3, y4, y5):
        h = self.encode(x)

        logits2 = self.head2(h)

        h3 = torch.cat([h, self.y2_embed(y2)], dim=1)
        h3 = self.dropout(self.relu(self.head3_hidden(h3)))
        logits3 = self.head3(h3)

        h4 = torch.cat([h, self.y3_embed(y3)], dim=1)
        h4 = self.dropout(self.relu(self.head4_hidden(h4)))
        logits4 = self.head4(h4)

        h5 = torch.cat([h, self.y4_embed(y4)], dim=1)
        h5 = self.dropout(self.relu(self.head5_hidden(h5)))
        logits5 = self.head5(h5)

        h6 = torch.cat([h, self.y5_embed(y5)], dim=1)
        h6 = self.dropout(self.relu(self.head6_hidden(h6)))
        logits6 = self.head6(h6)

        return logits2, logits3, logits4, logits5, logits6


def load_npz_dict(path):
    obj = np.load(path)
    return {k: obj[k] for k in obj.files}


def apply_mask(logits, allowed_mask):
    very_neg = torch.full_like(logits, -1e9)
    return torch.where(allowed_mask, logits, very_neg)


def strict_decode(model, x, mask23, mask34, mask45, mask56):
    h = model.encode(x)

    logits2 = model.head2(h)
    pred2 = torch.argmax(logits2, dim=1)

    h3 = torch.cat([h, model.y2_embed(pred2)], dim=1)
    h3 = model.dropout(model.relu(model.head3_hidden(h3)))
    logits3 = model.head3(h3)
    allowed3 = torch.index_select(mask23, 0, pred2)
    logits3 = apply_mask(logits3, allowed3)
    pred3 = torch.argmax(logits3, dim=1)

    h4 = torch.cat([h, model.y3_embed(pred3)], dim=1)
    h4 = model.dropout(model.relu(model.head4_hidden(h4)))
    logits4 = model.head4(h4)
    allowed4 = torch.index_select(mask34, 0, pred3)
    logits4 = apply_mask(logits4, allowed4)
    pred4 = torch.argmax(logits4, dim=1)

    h5 = torch.cat([h, model.y4_embed(pred4)], dim=1)
    h5 = model.dropout(model.relu(model.head5_hidden(h5)))
    logits5 = model.head5(h5)
    allowed5 = torch.index_select(mask45, 0, pred4)
    logits5 = apply_mask(logits5, allowed5)
    pred5 = torch.argmax(logits5, dim=1)

    h6 = torch.cat([h, model.y5_embed(pred5)], dim=1)
    h6 = model.dropout(model.relu(model.head6_hidden(h6)))
    logits6 = model.head6(h6)
    allowed6 = torch.index_select(mask56, 0, pred5)
    logits6 = apply_mask(logits6, allowed6)
    pred6 = torch.argmax(logits6, dim=1)

    return {
        "pred2": pred2,
        "pred3": pred3,
        "pred4": pred4,
        "pred5": pred5,
        "pred6": pred6,
        "logits6": logits6,
    }


def evaluate(model, loader, mask23, mask34, mask45, mask56, loss_weights, criterion):
    model.eval()

    total_loss = 0.0
    total_n = 0

    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    correct6 = 0
    top5_6 = 0

    with torch.no_grad():
        for batch in loader:
            x, y2, y3, y4, y5, y6 = batch

            x = x.to(DEVICE)
            y2 = y2.to(DEVICE)
            y3 = y3.to(DEVICE)
            y4 = y4.to(DEVICE)
            y5 = y5.to(DEVICE)
            y6 = y6.to(DEVICE)

            logits2, logits3, logits4, logits5, logits6 = model.forward_teacher_forced(x, y2, y3, y4, y5)

            allowed3 = torch.index_select(mask23, 0, y2)
            allowed4 = torch.index_select(mask34, 0, y3)
            allowed5 = torch.index_select(mask45, 0, y4)
            allowed6 = torch.index_select(mask56, 0, y5)

            logits3 = apply_mask(logits3, allowed3)
            logits4 = apply_mask(logits4, allowed4)
            logits5 = apply_mask(logits5, allowed5)
            logits6 = apply_mask(logits6, allowed6)

            loss = (
                loss_weights["y2"] * criterion(logits2, y2) +
                loss_weights["y3"] * criterion(logits3, y3) +
                loss_weights["y4"] * criterion(logits4, y4) +
                loss_weights["y5"] * criterion(logits5, y5) +
                loss_weights["y6"] * criterion(logits6, y6)
            )

            strict = strict_decode(model, x, mask23, mask34, mask45, mask56)

            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n

            correct2 += (strict["pred2"] == y2).sum().item()
            correct3 += (strict["pred3"] == y3).sum().item()
            correct4 += (strict["pred4"] == y4).sum().item()
            correct5 += (strict["pred5"] == y5).sum().item()
            correct6 += (strict["pred6"] == y6).sum().item()

            k = min(5, strict["logits6"].shape[1])
            topk = torch.topk(strict["logits6"], k=k, dim=1)[1]
            top5_6 += topk.eq(y6.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "loss": total_loss / total_n,
        "acc_y2": correct2 / total_n,
        "acc_y3": correct3 / total_n,
        "acc_y4": correct4 / total_n,
        "acc_y5": correct5 / total_n,
        "acc_y6": correct6 / total_n,
        "top5_y6": top5_6 / total_n,
    }


def main():
    print("entered main", flush=True)

    project_dir = Path(__file__).resolve().parents[2]

    processed_dir = project_dir / "data" / "processed"
    artifacts_dir = project_dir / "training" / "artifacts"
    hierarchy_dir = artifacts_dir / "hierarchy"
    label_maps_dir = artifacts_dir / "label_maps"
    embedder_dir = artifacts_dir / "embedder"
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train_embed.npy")
    X_valid = np.load(processed_dir / "X_valid_embed.npy")
    X_test = np.load(processed_dir / "X_test_embed.npy")
    print("loaded X arrays", X_train.shape, X_valid.shape, X_test.shape, flush=True)

    y_train = load_npz_dict(processed_dir / "y_train_embed.npz")
    y_valid = load_npz_dict(processed_dir / "y_valid_embed.npz")
    y_test = load_npz_dict(processed_dir / "y_test_embed.npz")
    print("loaded y arrays", flush=True)

    with open(hierarchy_dir / "hierarchy_embed.pkl", "rb") as f:
        hierarchy = pickle.load(f)
    print("loaded hierarchy", flush=True)

    with open(label_maps_dir / "label_maps_embed.pkl", "rb") as f:
        label_maps = pickle.load(f)
    print("loaded label maps", flush=True)

    with open(embedder_dir / "embed_metadata.pkl", "rb") as f:
        embed_metadata = pickle.load(f)
    print("loaded embed metadata", flush=True)

    input_dim = int(X_train.shape[1])
    n2 = len(label_maps["y2"]["classes"])
    n3 = len(label_maps["y3"]["classes"])
    n4 = len(label_maps["y4"]["classes"])
    n5 = len(label_maps["y5"]["classes"])
    n6 = len(label_maps["y6"]["classes"])

    train_ds = HierEmbedDataset(X_train, y_train)
    valid_ds = HierEmbedDataset(X_valid, y_valid)
    test_ds = HierEmbedDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print("built datasets and dataloaders", flush=True)

    mask23 = torch.tensor(hierarchy["mask23"], dtype=torch.bool).to(DEVICE)
    mask34 = torch.tensor(hierarchy["mask34"], dtype=torch.bool).to(DEVICE)
    mask45 = torch.tensor(hierarchy["mask45"], dtype=torch.bool).to(DEVICE)
    mask56 = torch.tensor(hierarchy["mask56"], dtype=torch.bool).to(DEVICE)

    model = HierarchicalEmbedMLP(
        input_dim=input_dim,
        n2=n2,
        n3=n3,
        n4=n4,
        n5=n5,
        n6=n6,
    ).to(DEVICE)
    print("built model", flush=True)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    loss_weights = {
        "y2": 0.5,
        "y3": 0.75,
        "y4": 1.0,
        "y5": 1.25,
        "y6": 1.5,
    }

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

        for batch_idx, batch in enumerate(train_loader):
            x, y2, y3, y4, y5, y6 = batch

            x = x.to(DEVICE)
            y2 = y2.to(DEVICE)
            y3 = y3.to(DEVICE)
            y4 = y4.to(DEVICE)
            y5 = y5.to(DEVICE)
            y6 = y6.to(DEVICE)

            optimizer.zero_grad()

            logits2, logits3, logits4, logits5, logits6 = model.forward_teacher_forced(x, y2, y3, y4, y5)

            allowed3 = torch.index_select(mask23, 0, y2)
            allowed4 = torch.index_select(mask34, 0, y3)
            allowed5 = torch.index_select(mask45, 0, y4)
            allowed6 = torch.index_select(mask56, 0, y5)

            logits3 = apply_mask(logits3, allowed3)
            logits4 = apply_mask(logits4, allowed4)
            logits5 = apply_mask(logits5, allowed5)
            logits6 = apply_mask(logits6, allowed6)

            loss = (
                loss_weights["y2"] * criterion(logits2, y2) +
                loss_weights["y3"] * criterion(logits3, y3) +
                loss_weights["y4"] * criterion(logits4, y4) +
                loss_weights["y5"] * criterion(logits5, y5) +
                loss_weights["y6"] * criterion(logits6, y6)
            )

            loss.backward()
            optimizer.step()

            batch_n = x.size(0)
            running_loss += loss.item() * batch_n
            total_n += batch_n

            if batch_idx % 50 == 0:
                print(f"epoch {epoch} batch {batch_idx} loss {loss.item():.4f}", flush=True)

        train_loss = running_loss / total_n

        valid_metrics = evaluate(
            model,
            valid_loader,
            mask23,
            mask34,
            mask45,
            mask56,
            loss_weights,
            criterion,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_acc_y2": valid_metrics["acc_y2"],
            "valid_acc_y3": valid_metrics["acc_y3"],
            "valid_acc_y4": valid_metrics["acc_y4"],
            "valid_acc_y5": valid_metrics["acc_y5"],
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
            torch.save(model.state_dict(), models_dir / "hierarchical_embed_best.pt")
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

    model.load_state_dict(torch.load(models_dir / "hierarchical_embed_best.pt", map_location=DEVICE))

    print("evaluating test set", flush=True)
    test_metrics = evaluate(
        model,
        test_loader,
        mask23,
        mask34,
        mask45,
        mask56,
        loss_weights,
        criterion,
    )

    with open(models_dir / "hierarchical_embed_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(models_dir / "hierarchical_embed_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "hidden_dim": HIDDEN_DIM,
        "parent_embed_dim": PARENT_EMBED_DIM,
        "dropout": DROPOUT,
        "device": DEVICE,
        "embedder_model_name": embed_metadata["model_name"],
        "embedding_dim": embed_metadata["embedding_dim"],
    }

    with open(models_dir / "hierarchical_embed_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("done", flush=True)
    print("test metrics:", flush=True)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    print("script started", flush=True)
    main()