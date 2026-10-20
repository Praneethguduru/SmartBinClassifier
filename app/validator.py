"""Core check: photo + order -> PASS/FAIL per item and overall.
Reads ONLY the photo and the order. It never touches metadata or labels."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "resnet18_best.pt"
THRESHOLD = 0.5            # frozen on validation, before the test set was opened

catalogue = pd.read_csv(HERE / "catalogue.csv")
ASINS = list(catalogue["asin"])
NAMES = dict(zip(catalogue["asin"], catalogue["product_name"]))

_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
def _get_model():
    """Load the model once, then reuse it."""
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(ASINS))
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model = m.eval()
    return _model


def predict_counts(image):
    """Photo -> {asin: units found}. `image` is a file path or a PIL image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    x = _tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        raw = _get_model()(x)[0]
    counts = torch.where(raw >= THRESHOLD, raw.round().clamp(1, 5), torch.zeros_like(raw))
    return {a: int(c) for a, c in zip(ASINS, counts)}, {a: float(r) for a, r in zip(ASINS, raw)}


def validate(image, order):
    """order = {asin: quantity}. Returns a dict with one row per item and an overall result."""
    # --- input checks: clear errors instead of crashes ---
    if not order:
        raise ValueError("Please select at least one item for the order.")
    for asin, qty in order.items():
        if asin not in NAMES:
            raise ValueError(f"Unknown product: {asin}")
        if not isinstance(qty, int) or qty < 1:
            raise ValueError(f"Quantity for '{NAMES[asin]}' must be a whole number of 1 or more.")
    if image is None:
        raise ValueError("Please choose a bin photo.")

    found, raw = predict_counts(image)

    items = []
    for asin, qty in order.items():
        f = found[asin]
        if f >= qty:
            result, reason = "PASS", ""
        elif f == 0:
            result, reason = "FAIL", "item not found"
        else:
            result, reason = "FAIL", f"fewer units found than ordered ({f} found, {qty} ordered)"
        items.append({"asin": asin, "product": NAMES[asin], "ordered": qty,
                      "found": f, "raw": round(raw[asin], 2),
                      "result": result, "reason": reason})

    failed = [i for i in items if i["result"] == "FAIL"]
    overall = "PASS" if not failed else "FAIL"
    summary = ("Every ordered item was found in the required quantity." if not failed else
               "; ".join(f"{i['product'][:40]}: {i['reason']}" for i in failed))
    return {"items": items, "overall": overall, "summary": summary}