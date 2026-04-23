import torch
from load_artifacts import load_artifacts


def predict_one(company_description):
    artifacts = load_artifacts()

    embedder = artifacts["embedder"]
    model = artifacts["model"]
    label_maps = artifacts["label_maps"]
    device = artifacts["device"]

    emb = embedder.encode(
        [company_description],
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    x = torch.tensor(emb, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs[0].cpu().numpy()
    pred_idx = int(probs.argmax(dim=1)[0].cpu().item())
    pred_prob_y6 = float(torch.max(probs, dim=1).values[0].cpu().item())

    pred_y6 = label_maps["y6"]["to_value"][pred_idx]
    pred_y2 = pred_y6[:2]
    pred_y3 = pred_y6[:3]
    pred_y4 = pred_y6[:4]
    pred_y5 = pred_y6[:5]

    
    y6_title_lookup = artifacts["y6_title_lookup"]
    pred_y6_title = y6_title_lookup.get(pred_y6, "")

    top_idx = probs_np.argsort()[::-1]

    pred_top5_y6 = []
    for i in top_idx:
        prob = float(probs_np[i])

        if prob < 1e-6:
            continue

        code = label_maps["y6"]["to_value"][int(i)]
        pred_top5_y6.append({
            "code": code,
            "title": y6_title_lookup.get(code, ""),
            "prob": prob,
        })

        if len(pred_top5_y6) == 5:
            break

    return {
        "pred_y2": pred_y2,
        "pred_y3": pred_y3,
        "pred_y4": pred_y4,
        "pred_y5": pred_y5,
        "pred_y6": pred_y6,
        "pred_y6_title": pred_y6_title,
        "pred_prob_y6": pred_prob_y6,
        "pred_top5_y6": pred_top5_y6,
    }