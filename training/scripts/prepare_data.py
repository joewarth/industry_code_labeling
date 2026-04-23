from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
VALID_SIZE = 0.15
TEST_SIZE = 0.15

NAICS_SAMPLE_N = 5000


def clean_text(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x == "":
        return None

    return x


def clean_naics_2022(x):
    if pd.isna(x):
        return None

    x = str(x)
    x = "".join(ch for ch in x if ch.isdigit())

    if x == "":
        return None

    x = x.zfill(6)

    if len(x) != 6:
        return None

    return x


def split_one_class(group, valid_size=VALID_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    n = len(group)

    if n <= 2:
        return group, group.iloc[0:0].copy(), group.iloc[0:0].copy()

    if 3 <= n <= 5:
        train_part, valid_part = train_test_split(
            group,
            test_size=valid_size,
            random_state=random_state
        )
        return train_part, valid_part, group.iloc[0:0].copy()

    train_valid_part, test_part = train_test_split(
        group,
        test_size=test_size,
        random_state=random_state
    )

    valid_share_of_train_valid = valid_size / (1.0 - test_size)

    train_part, valid_part = train_test_split(
        train_valid_part,
        test_size=valid_share_of_train_valid,
        random_state=random_state
    )

    return train_part, valid_part, test_part


def prep_exio_df(df):
    keep_cols = [
        "Company Name",
        "Company Description",
        "2022 NAICS Code",
        "2022 NAICS Title",
    ]

    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep_cols].copy()

    rename_map = {
        "Company Name": "company_name",
        "Company Description": "company_description",
        "2022 NAICS Code": "naics_2022",
        "2022 NAICS Title": "naics_2022_title",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "company_name" not in df.columns:
        df["company_name"] = None

    if "naics_2022_title" not in df.columns:
        df["naics_2022_title"] = None

    df["company_description"] = df["company_description"].apply(clean_text)
    df["naics_2022"] = df["naics_2022"].apply(clean_naics_2022)

    df["company_name"] = df["company_name"].apply(
        lambda x: None if pd.isna(x) else str(x).strip()
    )
    df["naics_2022_title"] = df["naics_2022_title"].apply(
        lambda x: None if pd.isna(x) else str(x).strip()
    )

    df["data_source"] = "exionaics"

    return df[[
        "company_name",
        "company_description",
        "naics_2022",
        "naics_2022_title",
        "data_source",
    ]].copy()


def prep_supplement_df(df):
    keep_cols = ["naics_code", "naics_title", "naics_description"]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep_cols].copy()

    rename_map = {
        "naics_code": "naics_2022",
        "naics_title": "naics_2022_title",
        "naics_description": "company_description",
    }
    df = df.rename(columns=rename_map)

    df["company_name"] = None
    df["company_description"] = df["company_description"].apply(clean_text)
    df["naics_2022"] = df["naics_2022"].apply(clean_naics_2022)
    df["naics_2022_title"] = df["naics_2022_title"].apply(
        lambda x: None if pd.isna(x) else str(x).strip()
    )

    # only true full 6-digit codes
    df = df.dropna(subset=["company_description", "naics_2022"]).copy()
    df = df[df["naics_2022"].str.fullmatch(r"\d{6}")].copy()

    df["data_source"] = "naics_supplement"

    return df[[
        "company_name",
        "company_description",
        "naics_2022",
        "naics_2022_title",
        "data_source",
    ]].copy()


def main():
    project_dir = Path(__file__).resolve().parents[2]

    raw_dir = project_dir / "data" / "raw"
    exio_path = raw_dir / "exionaics_raw.csv"
    supplement_path = raw_dir / "2022_naics_supplemental.xlsx"

    interim_dir = project_dir / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    exio_df = pd.read_csv(exio_path)
    exio_df = prep_exio_df(exio_df)
    exio_df = exio_df.dropna(subset=["company_description", "naics_2022"]).copy()

    supplement_sample = pd.DataFrame(columns=exio_df.columns)

    if NAICS_SAMPLE_N > 0 and supplement_path.exists():
        supplement_df = pd.read_excel(supplement_path)
        supplement_df = prep_supplement_df(supplement_df)

        sample_n = min(NAICS_SAMPLE_N, len(supplement_df))
        supplement_sample = supplement_df.sample(
            n=NAICS_SAMPLE_N,
            replace=True,
            random_state=RANDOM_STATE
        ).copy()

        df = pd.concat([exio_df, supplement_sample], ignore_index=True)
    else:
        df = exio_df.copy()

    df["y2"] = df["naics_2022"].str[:2]
    df["y3"] = df["naics_2022"].str[:3]
    df["y4"] = df["naics_2022"].str[:4]
    df["y5"] = df["naics_2022"].str[:5]
    df["y6"] = df["naics_2022"]

    df = df.drop_duplicates(subset=["company_description", "naics_2022"]).copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    class_counts = df["y6"].value_counts().sort_index()

    train_parts = []
    valid_parts = []
    test_parts = []

    for y6_value, group in df.groupby("y6", sort=True):
        train_part, valid_part, test_part = split_one_class(group)
        train_parts.append(train_part)
        valid_parts.append(valid_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_y6 = set(train_df["y6"].unique())
    valid_y6 = set(valid_df["y6"].unique())
    test_y6 = set(test_df["y6"].unique())

    missing_valid = sorted(valid_y6 - train_y6)
    missing_test = sorted(test_y6 - train_y6)

    if missing_valid:
        raise ValueError(f"Validation contains y6 classes not in training: {missing_valid[:10]}")

    if missing_test:
        raise ValueError(f"Test contains y6 classes not in training: {missing_test[:10]}")

    cleaned_path = interim_dir / "exionaics_2022_clean.csv"
    train_path = interim_dir / "train.csv"
    valid_path = interim_dir / "valid.csv"
    test_path = interim_dir / "test.csv"
    counts_path = interim_dir / "y6_class_counts.csv"

    df.to_csv(cleaned_path, index=False)
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    class_counts.rename_axis("y6").reset_index(name="count").to_csv(counts_path, index=False)

    print(f"Cleaned full dataset saved to: {cleaned_path}")
    print(f"Train saved to: {train_path}")
    print(f"Valid saved to: {valid_path}")
    print(f"Test saved to: {test_path}")
    print(f"Class counts saved to: {counts_path}")

    print("\nShapes:")
    print(f"ExioNAICS rows:          {exio_df.shape}")
    print(f"Supplement sampled rows: {supplement_sample.shape}")
    print(f"Full:                    {df.shape}")
    print(f"Train:                   {train_df.shape}")
    print(f"Valid:                   {valid_df.shape}")
    print(f"Test:                    {test_df.shape}")

    print("\nUnique y6 counts:")
    print(f"Full:  {df['y6'].nunique()}")
    print(f"Train: {train_df['y6'].nunique()}")
    print(f"Valid: {valid_df['y6'].nunique()}")
    print(f"Test:  {test_df['y6'].nunique()}")

    print("\nOverlap checks:")
    print(f"Valid y6 missing from train: {len(missing_valid)}")
    print(f"Test y6 missing from train:  {len(missing_test)}")

    print("\nData source counts:")
    print(df["data_source"].value_counts(dropna=False))

    print("\nSample rows:")
    print(train_df[[
        "company_description", "naics_2022", "y2", "y3", "y4", "y5", "y6", "data_source"
    ]].head())


if __name__ == "__main__":
    main()