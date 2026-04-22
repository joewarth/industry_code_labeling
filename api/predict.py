import torch
from load_artifacts import load_artifacts


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
    top5_idx = torch.topk(logits6, k=k, dim=1).indices

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


def predict_one(company_description):
    artifacts = load_artifacts()

    embedder = artifacts["embedder"]
    model = artifacts["model"]
    label_maps = artifacts["label_maps"]
    mask23 = artifacts["mask23"]
    mask34 = artifacts["mask34"]
    mask45 = artifacts["mask45"]
    mask56 = artifacts["mask56"]
    device = artifacts["device"]

    emb = embedder.encode(
        [company_description],
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    x = torch.tensor(emb, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = strict_decode(model, x, mask23, mask34, mask45, mask56)

    pred2_idx = int(out["pred2"][0].cpu().item())
    pred3_idx = int(out["pred3"][0].cpu().item())
    pred4_idx = int(out["pred4"][0].cpu().item())
    pred5_idx = int(out["pred5"][0].cpu().item())
    pred6_idx = int(out["pred6"][0].cpu().item())

    pred_y2 = label_maps["y2"]["to_value"][pred2_idx]
    pred_y3 = label_maps["y3"]["to_value"][pred3_idx]
    pred_y4 = label_maps["y4"]["to_value"][pred4_idx]
    pred_y5 = label_maps["y5"]["to_value"][pred5_idx]
    pred_y6 = label_maps["y6"]["to_value"][pred6_idx]

    pred_prob_y2 = float(torch.max(out["probs2"], dim=1).values[0].cpu().item())
    pred_prob_y3 = float(torch.max(out["probs3"], dim=1).values[0].cpu().item())
    pred_prob_y4 = float(torch.max(out["probs4"], dim=1).values[0].cpu().item())
    pred_prob_y5 = float(torch.max(out["probs5"], dim=1).values[0].cpu().item())
    pred_prob_y6 = float(torch.max(out["probs6"], dim=1).values[0].cpu().item())

    top5_idx = out["top5_idx"][0].cpu().numpy().tolist()
    pred_top5_y6 = [label_maps["y6"]["to_value"][int(i)] for i in top5_idx]

    return {
        "pred_y2": pred_y2,
        "pred_y3": pred_y3,
        "pred_y4": pred_y4,
        "pred_y5": pred_y5,
        "pred_y6": pred_y6,
        "pred_prob_y2": pred_prob_y2,
        "pred_prob_y3": pred_prob_y3,
        "pred_prob_y4": pred_prob_y4,
        "pred_prob_y5": pred_prob_y5,
        "pred_prob_y6": pred_prob_y6,
        "pred_top5_y6": pred_top5_y6,
    }