import time, json, csv
from pathlib import Path

import streamlit as st
from PIL import Image

from validator import validate, catalogue, NAMES
from bin_index import load_pool, bins_matching, photo_path, DATA

LOG = Path(__file__).resolve().parent.parent / "logs" / "validation_log.csv"
LOG.parent.mkdir(exist_ok=True)

st.set_page_config(page_title="Smart Bin Classifier", layout="wide")
st.title("Smart Bin Classifier")

@st.cache_data
def get_pool():
    return load_pool()
pool = get_pool()

def log_request(order, photo_id, result, seconds):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "order", "photo_id", "model_output", "result", "seconds"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(order), photo_id,
                    json.dumps({i["asin"]: i["found"] for i in result["items"]}),
                    result["overall"], round(seconds, 3)])

def show_result(result):
    for i in result["items"]:
        c = st.columns([5, 1, 1, 1])
        c[0].write(i["product"][:60])
        c[1].write(f"Ordered: {i['ordered']}")
        c[2].write(f"Found: {i['found']}")
        (c[3].success if i["result"] == "PASS" else c[3].error)(i["result"])
    if result["overall"] == "PASS":
        st.success("ORDER PASS: " + result["summary"])
    else:
        st.error("ORDER FAIL: " + result["summary"])

name_to_asin = {NAMES[a][:70]: a for a in catalogue["asin"]}

mode = st.radio("Mode", ["Normal", "Blind"], horizontal=True)

# ---- order builder ----
st.subheader("1. Build the order")
chosen = st.multiselect("Products", list(name_to_asin))
order = {}
for n in chosen:
    order[name_to_asin[n]] = int(st.number_input(f"Quantity: {n}", 1, 5, 1, key=n))

if mode == "Normal":
    # ---- matching bin from metadata (the ONLY place metadata is used) ----
    st.subheader("2. Bin photo")
    ok_order = bool(order)
    matches = bins_matching(pool, order) if ok_order else pool.iloc[0:0]
    all_ids = list(pool.index)

    use_other = st.checkbox("Choose a different bin (to show a FAIL)")
    if use_other:
        bin_id = st.selectbox("Pick any bin", all_ids[:500])
    elif len(matches):
        bin_id = st.selectbox("Matching bin", list(matches.index[:100]))
    else:
        bin_id = None
        if ok_order:
            st.warning("No demo bin holds this order (3 or more units of one product is rare). Lower the quantity or tick 'Choose a different bin'.")

    image = None
    if bin_id is not None:
        path = photo_path(pool, bin_id)
        image = Image.open(path)
        st.image(image, width=350, caption=bin_id)
        photo_id = bin_id
else:
    st.subheader("2. Blind bin")
    blind_ids = sorted(p.name for p in (DATA / "blind").glob("*.jpg"))
    typed = st.text_input("Type a bin id from the blind folder (for example 123 or 123.jpg)").strip()
    if typed and not typed.lower().endswith(".jpg"):
        typed += ".jpg"
    image, photo_id = None, typed
    if typed:
        if typed not in blind_ids:
            st.error(f"'{typed}' is not in the blind folder.")
        else:
            image = Image.open(DATA / "blind" / typed)
            st.image(image, width=350, caption=typed)

# ---- validate ----
st.subheader("3. Validate")
if st.button("Validate", type="primary"):
    t0 = time.time()
    try:
        result = validate(image, order)          # photo + order only
        log_request(order, photo_id if image is not None else "", result, time.time() - t0)
        show_result(result)
    except ValueError as e:
        st.error(str(e))

if LOG.exists():
    st.download_button("Download log", LOG.read_bytes(), "validation_log.csv")