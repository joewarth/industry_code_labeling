from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


MAX_WORDS = 30000
MAX_LEN = 100
OOV_TOKEN = "[OOV]"


def build_label_map(values):
    classes = sorted(pd.Series(values).astype(str).unique())
    to_index = {v: i for i, v in enumerate(classes)}
    to_value = {i: v for i, v in enumerate(classes)}
    return {
        "classes": classes,
        "to_index": to_index,
        "to_value": to_value,
    }


def encode_labels(values, label_map):
    return np.array([label_map["to_index"][str(v)] for v in values], dtype=np.int32)


def build_hierarchy_masks(train_df, label_maps):
    y2_to_idx = label_maps["y2"]["to_index"]
    y3_to_idx = label_maps["y3"]["to_index"]
    y4_to_idx = label_maps["y4"]["to_index"]
    y5_to_idx = label_maps["y5"]["to_index"]
    y6_to_idx = label_maps["y6"]["to_index"]

    mask23 = np.zeros((len(y2_to_idx), len(y3_to_idx)), dtype=bool)
    mask34 = np.zeros((len(y3_to_idx), len(y4_to_idx)), dtype=bool)
    mask45 = np.zeros((len(y4_to_idx), len(y5_to_idx)), dtype=bool)
    mask56 = np.zeros((len(y5_to_idx), len(y6_to_idx)), dtype=bool)

    for row in train_df[["y2", "y3", "y4", "y5", "y6"]].drop_duplicates().itertuples(index=False):
        y2, y3, y4, y5, y6 = map(str, row)

        mask23[y2_to_idx[y2], y3_to_idx[y3]] = True
        mask34[y3_to_idx[y3], y4_to_idx[y4]] = True
        mask45[y4_to_idx[y4], y5_to_idx[y5]] = True
        mask56[y5_to_idx[y5], y6_to_idx[y6]] = True

    return {
        "mask23": mask23,
        "mask34": mask34,
        "mask45": mask45,
        "mask56": mask56,
    }


def check_split_labels(split_df, split_name, label_maps):
    for level in ["y2", "y3", "y4", "y5", "y6"]:
        unseen = sorted(set(split_df[level].astype(str).unique()) - set(label_maps[level]["classes"]))
        if unseen:
            raise ValueError(f"{split_name} contains unseen {level} labels: {unseen[:10]}")


def main():
    project_dir = Path(__file__).resolve().parents[2]

    interim_dir = project_dir / "data" / "interim"
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = project_dir / "training" / "artifacts"
    tokenizer_dir = artifacts_dir / "tokenizer"
    label_maps_dir = artifacts_dir / "label_maps"
    hierarchy_dir = artifacts_dir / "hierarchy"

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    label_maps_dir.mkdir(parents=True, exist_ok=True)
    hierarchy_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(interim_dir / "train.csv")
    valid_df = pd.read_csv(interim_dir / "valid.csv")
    test_df = pd.read_csv(interim_dir / "test.csv")

    # text
    X_train_text = train_df["company_description"].astype(str).tolist()
    X_valid_text = valid_df["company_description"].astype(str).tolist()
    X_test_text = test_df["company_description"].astype(str).tolist()

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_valid_seq = tokenizer.texts_to_sequences(X_valid_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_valid = pad_sequences(X_valid_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # label maps built from train only
    label_maps = {
        "y2": build_label_map(train_df["y2"]),
        "y3": build_label_map(train_df["y3"]),
        "y4": build_label_map(train_df["y4"]),
        "y5": build_label_map(train_df["y5"]),
        "y6": build_label_map(train_df["y6"]),
    }

    check_split_labels(valid_df, "valid", label_maps)
    check_split_labels(test_df, "test", label_maps)

    y_train = {
        "y2": encode_labels(train_df["y2"], label_maps["y2"]),
        "y3": encode_labels(train_df["y3"], label_maps["y3"]),
        "y4": encode_labels(train_df["y4"], label_maps["y4"]),
        "y5": encode_labels(train_df["y5"], label_maps["y5"]),
        "y6": encode_labels(train_df["y6"], label_maps["y6"]),
    }

    y_valid = {
        "y2": encode_labels(valid_df["y2"], label_maps["y2"]),
        "y3": encode_labels(valid_df["y3"], label_maps["y3"]),
        "y4": encode_labels(valid_df["y4"], label_maps["y4"]),
        "y5": encode_labels(valid_df["y5"], label_maps["y5"]),
        "y6": encode_labels(valid_df["y6"], label_maps["y6"]),
    }

    y_test = {
        "y2": encode_labels(test_df["y2"], label_maps["y2"]),
        "y3": encode_labels(test_df["y3"], label_maps["y3"]),
        "y4": encode_labels(test_df["y4"], label_maps["y4"]),
        "y5": encode_labels(test_df["y5"], label_maps["y5"]),
        "y6": encode_labels(test_df["y6"], label_maps["y6"]),
    }

    hierarchy = build_hierarchy_masks(train_df, label_maps)

    # save arrays
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_valid.npy", X_valid)
    np.save(processed_dir / "X_test.npy", X_test)

    np.savez(processed_dir / "y_train.npz", **y_train)
    np.savez(processed_dir / "y_valid.npz", **y_valid)
    np.savez(processed_dir / "y_test.npz", **y_test)

    # optional: save raw text and codes for later inspection/evaluation
    train_df.to_csv(processed_dir / "train_processed_reference.csv", index=False)
    valid_df.to_csv(processed_dir / "valid_processed_reference.csv", index=False)
    test_df.to_csv(processed_dir / "test_processed_reference.csv", index=False)

    # save artifacts
    with open(tokenizer_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(label_maps_dir / "label_maps.pkl", "wb") as f:
        pickle.dump(label_maps, f)

    with open(hierarchy_dir / "hierarchy.pkl", "wb") as f:
        pickle.dump(hierarchy, f)

    metadata = {
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "oov_token": OOV_TOKEN,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "vocab_size_raw": len(tokenizer.word_index),
        "vocab_size_capped": min(MAX_WORDS, len(tokenizer.word_index) + 1),
        "n_classes_y2": len(label_maps["y2"]["classes"]),
        "n_classes_y3": len(label_maps["y3"]["classes"]),
        "n_classes_y4": len(label_maps["y4"]["classes"]),
        "n_classes_y5": len(label_maps["y5"]["classes"]),
        "n_classes_y6": len(label_maps["y6"]["classes"]),
    }

    with open(processed_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Saved tokenized arrays:")
    print(processed_dir / "X_train.npy")
    print(processed_dir / "X_valid.npy")
    print(processed_dir / "X_test.npy")

    print("\nSaved label arrays:")
    print(processed_dir / "y_train.npz")
    print(processed_dir / "y_valid.npz")
    print(processed_dir / "y_test.npz")

    print("\nSaved artifacts:")
    print(tokenizer_dir / "tokenizer.pkl")
    print(label_maps_dir / "label_maps.pkl")
    print(hierarchy_dir / "hierarchy.pkl")

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test: ", X_test.shape)

    print("\nClass counts:")
    print("y2:", len(label_maps["y2"]["classes"]))
    print("y3:", len(label_maps["y3"]["classes"]))
    print("y4:", len(label_maps["y4"]["classes"]))
    print("y5:", len(label_maps["y5"]["classes"]))
    print("y6:", len(label_maps["y6"]["classes"]))

    print("\nTokenizer:")
    print("Raw vocab size:", len(tokenizer.word_index))
    print("Capped vocab size:", min(MAX_WORDS, len(tokenizer.word_index) + 1))


if __name__ == "__main__":
    main()
