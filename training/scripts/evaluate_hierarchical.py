from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 64
EMBED_DIM = 128
NUM_FILTERS = 128
HIDDEN_DIM = 256
PARENT_EMBED_DIM = 32
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(1)


class HierDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.tensor(X, dtype=torch.long)
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


class HierarchicalCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        n2,
        n3,
        n4,
        n5,
        n6,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        hidden_dim=HIDDEN_DIM,
        parent_embed_dim=PARENT_EMBED_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5)

        encoder_dim = num_filters * 3

        self.encoder_fc = nn.Linear(encoder_dim, hidden_dim)
        self.encoder_dropout = nn.Dropout(dropout)

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
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)

        c3 = self.relu(self.conv3(emb))
        c3 = torch.max(c3, dim=2)[0]

        c4 = self.relu(self.conv4(emb))
        c4 = torch.max(c4, dim=2)[0]

        c5 = self.relu(self.conv5(emb))
        c5 = torch.max(c5, dim=2)[0]

        h = torch.cat([c3, c4, c5], dim=1)
        h = self.relu(self.encoder_fc(h))
        h = self.encoder_dropout(h)
        return h


def load_npz_dict(path):
    obj = np.load(path)
    return {k: obj[k] for k in obj.files}


def apply_mask(logits, allowed_mask):
    very_neg = torch.full_like(logits, -1e9)
    return torch.where(allowed_mask, logits, very_neg)


def strict_decode(model, x, mask23, mask34, mask45, mask56):
    h = model.encode(x)

    logits2 = model.head2(h)
    probs2 = torch.softmax(logits2, dim=1)
    pred2 = torch.argmax(logits2, dim=1)

    h3 = torch.cat([h, model.y2_embed(pred2)], dim=1)
    h3 = model.dropout(model.relu(model.head3_hidden(h3)))
    logits3 = model.head3(h3)
    allowed3 = torch.index_select(mask23, 0, pred2)
    logits3 = apply_mask(logits3, allowed3)
    probs3 = torch.softmax(logits3, dim=1)
    pred3 = torch.argmax(logits3, dim=1)

    h4 = torch.cat([h, model.y3_embed(pred3)], dim=1)
    h4 = model.dropout(model.relu(model.head4_hidden(h4)))
    logits4 = model.head4(h4)
    allowed4 = torch.index_select(mask34, 0, pred3)
    logits4 = apply_mask(logits4, allowed4)
    probs4 = torch.softmax(logits4, dim=1)
    pred4 = torch.argmax(logits4, dim=1)

    h5 = torch.cat([h, model.y4_embed(pred4)], dim=1)
    h5 = model.dropout(model.relu(model.head5_hidden(h5)))
    logits5 = model.head5(h5)
    allowed5 = torch.index_select(mask45, 0, pred4)
    logits5 = apply_mask(logits5, allowed5)
    probs5 = torch.softmax(logits5, dim=1)
    pred5 = torch.argmax(logits5, dim=1)

    h6 = torch.cat([h, model.y5_embed(pred5)], dim=1)
    h6 = model.dropout(model.relu(model.head6_hidden(h6)))
    logits6 = model.head6(h6)
    allowed6 = torch.index_select(mask56, 0, pred5)
    logits6 = apply_mask(logits6, allowed6)
    probs6 = torch.softmax(logits6, dim=1)
    pred6 = torch.argmax(logits6, dim=1)

    k = min(5, logits6.shape[1])
    top5_idx = torch.topk(logits6, k=k, dim=1)[1]

    return {
        "pred2": pred2,
        "pred3": pred3,
        "pred4": pred4,
        "pred5": pred5,
        "pred6": pred6,
        "probs2": probs2,
        "probs3": probs3,
        "probs4": probs4,
        "probs5": probs5,
        "probs6": probs6,
        "top5_idx": top5_idx,
    }


def idx_to_code(arr, label_map):
    return [label_map["to_value"][int(x)] for x in arr]


def main():
    print("starting evaluation", flush=True)

    project_dir = Path(__file__).resolve().parents[2]

    processed_dir = project_dir / "data" / "processed"
    artifacts_dir = project_dir / "training" / "artifacts"
    hierarchy_dir = artifacts_dir / "hierarchy"
    label_maps_dir = artifacts_dir / "label_maps"
    tokenizer_dir = artifacts_dir / "tokenizer"
    models_dir = artifacts_dir / "models"
    predictions_dir = project_dir / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(processed_dir / "X_test.npy")
    y_test = load_npz_dict(processed_dir / "y_test.npz")
    test_ref = pd.read_csv(processed_dir / "test_processed_reference.csv")

    with open(hierarchy_dir / "hierarchy.pkl", "rb") as f:
        hierarchy = pickle.load(f)

    with open(label_maps_dir / "label_maps.pkl", "rb") as f:
        label_maps = pickle.load(f)

    with open(tokenizer_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
    n2 = len(label_maps["y2"]["classes"])
    n3 = len(label_maps["y3"]["classes"])
    n4 = len(label_maps["y4"]["classes"])
    n5 = len(label_maps["y5"]["classes"])
    n6 = len(label_maps["y6"]["classes"])

    test_ds = HierDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    mask23 = torch.tensor(hierarchy["mask23"], dtype=torch.bool).to(DEVICE)
    mask34 = torch.tensor(hierarchy["mask34"], dtype=torch.bool).to(DEVICE)
    mask45 = torch.tensor(hierarchy["mask45"], dtype=torch.bool).to(DEVICE)
    mask56 = torch.tensor(hierarchy["mask56"], dtype=torch.bool).to(DEVICE)

    model = HierarchicalCNN(
        vocab_size=vocab_size,
        n2=n2,
        n3=n3,
        n4=n4,
        n5=n5,
        n6=n6,
    ).to(DEVICE)

    model_path = models_dir / "hierarchical_cnn_best.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_pred2 = []
    all_pred3 = []
    all_pred4 = []
    all_pred5 = []
    all_pred6 = []
    all_prob2 = []
    all_prob3 = []
    all_prob4 = []
    all_prob5 = []
    all_prob6 = []
    all_top5 = []

    with torch.no_grad():
        for batch in test_loader:
            x, y2, y3, y4, y5, y6 = batch
            x = x.to(DEVICE)

            out = strict_decode(model, x, mask23, mask34, mask45, mask56)

            all_pred2.extend(out["pred2"].cpu().numpy().tolist())
            all_pred3.extend(out["pred3"].cpu().numpy().tolist())
            all_pred4.extend(out["pred4"].cpu().numpy().tolist())
            all_pred5.extend(out["pred5"].cpu().numpy().tolist())
            all_pred6.extend(out["pred6"].cpu().numpy().tolist())

            all_prob2.extend(torch.max(out["probs2"], dim=1)[0].cpu().numpy().tolist())
            all_prob3.extend(torch.max(out["probs3"], dim=1)[0].cpu().numpy().tolist())
            all_prob4.extend(torch.max(out["probs4"], dim=1)[0].cpu().numpy().tolist())
            all_prob5.extend(torch.max(out["probs5"], dim=1)[0].cpu().numpy().tolist())
            all_prob6.extend(torch.max(out["probs6"], dim=1)[0].cpu().numpy().tolist())

            all_top5.extend(out["top5_idx"].cpu().numpy().tolist())

    pred_y2 = idx_to_code(all_pred2, label_maps["y2"])
    pred_y3 = idx_to_code(all_pred3, label_maps["y3"])
    pred_y4 = idx_to_code(all_pred4, label_maps["y4"])
    pred_y5 = idx_to_code(all_pred5, label_maps["y5"])
    pred_y6 = idx_to_code(all_pred6, label_maps["y6"])

    pred_top5_y6_codes = []
    for row in all_top5:
        pred_top5_y6_codes.append(
            [label_maps["y6"]["to_value"][int(i)] for i in row]
        )

    out_df = test_ref.copy()

    out_df["pred_y2"] = pred_y2
    out_df["pred_y3"] = pred_y3
    out_df["pred_y4"] = pred_y4
    out_df["pred_y5"] = pred_y5
    out_df["pred_y6"] = pred_y6

    out_df["pred_prob_y2"] = all_prob2
    out_df["pred_prob_y3"] = all_prob3
    out_df["pred_prob_y4"] = all_prob4
    out_df["pred_prob_y5"] = all_prob5
    out_df["pred_prob_y6"] = all_prob6

    out_df["match_y2"] = (out_df["y2"].astype(str) == out_df["pred_y2"].astype(str)).astype(int)
    out_df["match_y3"] = (out_df["y3"].astype(str) == out_df["pred_y3"].astype(str)).astype(int)
    out_df["match_y4"] = (out_df["y4"].astype(str) == out_df["pred_y4"].astype(str)).astype(int)
    out_df["match_y5"] = (out_df["y5"].astype(str) == out_df["pred_y5"].astype(str)).astype(int)
    out_df["match_y6"] = (out_df["y6"].astype(str) == out_df["pred_y6"].astype(str)).astype(int)

    out_df["pred_top5_y6"] = [" | ".join(x) for x in pred_top5_y6_codes]
    out_df["top5_contains_true_y6"] = [
        int(str(true_y6) in top5_list)
        for true_y6, top5_list in zip(out_df["y6"].astype(str), pred_top5_y6_codes)
    ]

    out_path = predictions_dir / "hierarchical_test_predictions.csv"
    out_df.to_csv(out_path, index=False)

    summary_dir = predictions_dir / "grouped_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    group_levels = ["y2", "y3", "y4", "y5", "y6"]

    metric_cols = [
        "match_y2",
        "match_y3",
        "match_y4",
        "match_y5",
        "match_y6",
        "top5_contains_true_y6",
    ]

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

        grouped.to_csv(summary_dir / f"summary_by_{level}.csv", index=False)

    summary = {
        "acc_y2": float(out_df["match_y2"].mean()),
        "acc_y3": float(out_df["match_y3"].mean()),
        "acc_y4": float(out_df["match_y4"].mean()),
        "acc_y5": float(out_df["match_y5"].mean()),
        "acc_y6": float(out_df["match_y6"].mean()),
        "top5_y6": float(out_df["top5_contains_true_y6"].mean()),
        "n_test_rows": int(len(out_df)),
    }

    summary_path = predictions_dir / "hierarchical_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("saved predictions to:", out_path, flush=True)
    print("saved summary to:", summary_path, flush=True)
    print("", flush=True)
    print("summary metrics:", flush=True)
    for k, v in summary.items():
        if k == "n_test_rows":
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