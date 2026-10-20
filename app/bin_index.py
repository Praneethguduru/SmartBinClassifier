from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "prepared"

def load_pool():
    """Return {asin: DataFrame of bins with its labels} from train + validation."""
    frames = []
    for grp in ["train", "validation"]:
        df = pd.read_csv(DATA / grp / "labels.csv", index_col=0)
        df["folder"] = grp
        frames.append(df)
    return pd.concat(frames)

def bins_matching(pool, order):
    """Bins that contain every ordered item in at least the ordered quantity."""
    ok = pd.Series(True, index=pool.index)
    for asin, qty in order.items():
        ok &= pool[asin] >= qty
    return pool[ok]

def photo_path(pool, bin_id):
    return DATA / pool.loc[bin_id, "folder"] / bin_id