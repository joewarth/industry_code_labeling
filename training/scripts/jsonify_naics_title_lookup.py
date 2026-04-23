from pathlib import Path
import pandas as pd
import json

project_dir = Path(".")
supplement_path = project_dir / "data" / "raw" / "2022_naics_supplemental.xlsx"
out_path = project_dir / "training" / "artifacts" / "label_maps" / "y6_title_lookup.json"

df = pd.read_excel(supplement_path)
df["naics_code"] = df["naics_code"].astype(str).str.strip()
df["naics_title"] = df["naics_title"].astype(str).str.strip()

lookup = dict(zip(df["naics_code"], df["naics_title"]))

with open(out_path, "w") as f:
    json.dump(lookup, f)

print(out_path)