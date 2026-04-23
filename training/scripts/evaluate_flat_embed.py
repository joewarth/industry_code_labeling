from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 64
HIDDEN_DIM = 768
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def main():
    print("starting evaluation", flush=True)

    project_dir = Path(__file__).resolve().parents[2]

    processed_dir = project_dir / "data" / "processed"
    artifacts_dir = project_dir / "training" / "artifacts"
    label_maps_dir = artifacts_dir / "label_maps"
    embedder_dir = artifacts_dir / "embedder"
    models_dir = artifacts_dir / "models"

    predictions_dir = project_dir / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    grouped_dir = predictions_dir / "grouped_summaries_flat"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(processed_dir / "X_test_embed.npy")
    y_test_obj = np.load(processed_dir / "y_test_embed.npz")
    y_test = y_test_obj["y6"]

    test_ref = pd.read_csv(processed_dir / "test_embed_reference.csv")

    with open(label_maps_dir / "label_maps_embed.pkl", "rb") as f:
        label_maps = pickle.load(f)

    with open(embedder_dir / "embed_metadata.pkl", "rb") as f:
        embed_metadata = pickle.load(f)

    input_dim = int(X_test.shape[1])
    n_classes = len(label_maps["y6"]["classes"])

    test_ds = FlatEmbedDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FlatEmbedMLP(
        input_dim=input_dim,
        n_classes=n_classes,
    ).to(DEVICE)

    model_path = models_dir / "flat_embed_best.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_pred_idx = []
    all_pred_prob = []
    all_top5_idx = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred_idx = torch.argmax(logits, dim=1)
            top5_idx = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            pred_prob = torch.max(probs, dim=1).values

            all_pred_idx.extend(pred_idx.cpu().numpy().tolist())
            all_pred_prob.extend(pred_prob.cpu().numpy().tolist())
            all_top5_idx.extend(top5_idx.cpu().numpy().tolist())

    pred_y6 = [label_maps["y6"]["to_value"][int(i)] for i in all_pred_idx]
    pred_top5_y6_codes = [
        [label_maps["y6"]["to_value"][int(i)] for i in row]
        for row in all_top5_idx
    ]

    out_df = test_ref.copy()

    out_df["pred_y6"] = pred_y6
    out_df["pred_prob_y6"] = all_pred_prob
    out_df["pred_top5_y6"] = [" | ".join(x) for x in pred_top5_y6_codes]

    # derive hierarchical prefixes from flat predictions
    out_df["pred_y2"] = out_df["pred_y6"].astype(str).str[:2]
    out_df["pred_y3"] = out_df["pred_y6"].astype(str).str[:3]
    out_df["pred_y4"] = out_df["pred_y6"].astype(str).str[:4]
    out_df["pred_y5"] = out_df["pred_y6"].astype(str).str[:5]

    # match flags
    out_df["match_y2"] = (out_df["y2"].astype(str) == out_df["pred_y2"].astype(str)).astype(int)
    out_df["match_y3"] = (out_df["y3"].astype(str) == out_df["pred_y3"].astype(str)).astype(int)
    out_df["match_y4"] = (out_df["y4"].astype(str) == out_df["pred_y4"].astype(str)).astype(int)
    out_df["match_y5"] = (out_df["y5"].astype(str) == out_df["pred_y5"].astype(str)).astype(int)
    out_df["match_y6"] = (out_df["y6"].astype(str) == out_df["pred_y6"].astype(str)).astype(int)

    out_df["top5_contains_true_y6"] = [
        int(str(true_y6) in top5_list)
        for true_y6, top5_list in zip(out_df["y6"].astype(str), pred_top5_y6_codes)
    ]

    row_path = predictions_dir / "flat_embed_test_predictions.csv"
    out_df.to_csv(row_path, index=False)

    summary = {
        "embedder_model_name": embed_metadata["model_name"],
        "embedding_dim": int(embed_metadata["embedding_dim"]),
        "acc_y2": float(out_df["match_y2"].mean()),
        "acc_y3": float(out_df["match_y3"].mean()),
        "acc_y4": float(out_df["match_y4"].mean()),
        "acc_y5": float(out_df["match_y5"].mean()),
        "acc_y6": float(out_df["match_y6"].mean()),
        "top5_y6": float(out_df["top5_contains_true_y6"].mean()),
        "n_test_rows": int(len(out_df)),
    }

    summary_path = predictions_dir / "flat_embed_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    group_levels = ["y2", "y3", "y4", "y5", "y6"]

    for level in group_levels:
        grouped = (
            out_df
            .groupby(level, dropna=False)
            .agg(
                n_obs=(level, "size"),
                match_y2_rate=("match_y2", "mean"),
                match_y3_rate=("match_y3", "mean"),
                match_y4_rate=("match_y4", "mean"),
                match_y5_rate=("match_y5", "mean"),
                match_y6_rate=("match_y6", "mean"),
                top5_contains_true_y6_rate=("top5_contains_true_y6", "mean"),
            )
            .reset_index()
            .sort_values(["n_obs", level], ascending=[False, True])
        )

        rate_cols = [
            "match_y2_rate",
            "match_y3_rate",
            "match_y4_rate",
            "match_y5_rate",
            "match_y6_rate",
            "top5_contains_true_y6_rate",
        ]
        grouped[rate_cols] = grouped[rate_cols].round(4)

        grouped.to_csv(grouped_dir / f"summary_by_{level}.csv", index=False)

        grouped_top5 = (
            out_df
            .groupby(level, dropna=False)
            .agg(
                n_obs=(level, "size"),
                top5_contains_true_y6_rate=("top5_contains_true_y6", "mean"),
            )
            .reset_index()
            .sort_values(["n_obs", level], ascending=[False, True])
        )
        grouped_top5["top5_contains_true_y6_rate"] = grouped_top5["top5_contains_true_y6_rate"].round(4)
        grouped_top5.to_csv(grouped_dir / f"top5_summary_by_{level}.csv", index=False)

    print("saved row-level predictions to:", row_path, flush=True)
    print("saved summary to:", summary_path, flush=True)
    print("saved grouped summaries to:", grouped_dir, flush=True)
    print("", flush=True)
    print("summary metrics:", flush=True)
    for k, v in summary.items():
        if k in ["embedder_model_name", "n_test_rows", "embedding_dim"]:
            print(f"{k}: {v}", flush=True)
        else:
            print(f"{k}: {v:.4f}", flush=True)

    print("", flush=True)
    print("sample errors:", flush=True)
    err_df = out_df[out_df["match_y6"] == 0].head(10)
    show_cols = [
        "company_name",
        "naics_2022",
        "pred_y6",
        "y2",
        "pred_y2",
        "y3",
        "pred_y3",
        "company_description",
    ]
    show_cols = [c for c in show_cols if c in err_df.columns]
    print(err_df[show_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()