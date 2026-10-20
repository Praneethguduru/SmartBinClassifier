"""Copy a SMALL demo sample (not the full data) into sample/ for the deployed app."""
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data" / "prepared", ROOT / "sample"
PER_PRODUCT = 10

frames = []
for grp in ["train", "validation"]:
    df = pd.read_csv(SRC / grp / "labels.csv", index_col=0)
    df["folder"] = grp
    frames.append(df)
pool = pd.concat(frames)
asins = [c for c in pool.columns if c != "folder"]

# 10 bins per product (fixed seed), skipping bins already chosen
chosen = set()
for a in asins:
    # first, bins with 2 or more units of this product (rare, so keep them all, up to 8)
    multi = pool.index[(pool[a] >= 2) & ~pool.index.isin(chosen)]
    chosen.update(pd.Series(multi).sample(min(8, len(multi)), random_state=0))
    # then fill up with ordinary bins that hold the product
    ids = pool.index[(pool[a] > 0) & ~pool.index.isin(chosen)]
    need = PER_PRODUCT - int((pool.loc[list(chosen), a] > 0).sum())
    if need > 0:
        chosen.update(pd.Series(ids).sample(min(need, len(ids)), random_state=0))

sample = pool.loc[sorted(chosen)]
for grp in ["train", "validation"]:
    part = sample[sample["folder"] == grp].drop(columns="folder")
    (DST / grp).mkdir(parents=True, exist_ok=True)
    part.to_csv(DST / grp / "labels.csv")
    for b in part.index:
        shutil.copy(SRC / grp / b, DST / grp / b)

# blind: photos only, no labels
(DST / "blind").mkdir(parents=True, exist_ok=True)
for p in (SRC / "blind").glob("*.jpg"):
    shutil.copy(p, DST / "blind" / p.name)

print("sample bins:", len(sample), "| blind photos:", len(list((DST / 'blind').glob('*.jpg'))))