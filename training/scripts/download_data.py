from pathlib import Path
from datasets import load_dataset

def main():
    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Yvnminc/ExioNAICS", split="train")
    df = ds.to_pandas()

    out_path = raw_dir / "exionaics_raw.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved raw data to: {out_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()