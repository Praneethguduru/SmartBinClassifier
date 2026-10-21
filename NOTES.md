cat >> NOTES.md << 'EOF'

## Split
Random split with seed 42 over the 2,000 bins: train 1,400 (70%), validation 200 (10%), test 300 (15%), blind 100 (5%). Every bin is in exactly one group. I hashed every photo (MD5) and found no duplicate image across groups. The test set was not touched until the final evaluation. Blind bins are photos only and their labels are never used.

## Catalogue check
I counted how many of the 2,000 bins each product appears in and kept every product that reaches at least 100. Exactly 15 products do, and the 16th has only 4, so the cut is clear. Each catalogue product appears in 119 to 177 bins.

## Answers from Satwik
- Using every product that appears in at least 100 of my 2,000 bins, with no cap: yes, based on my own frequencies, and document how the catalogue was built and checked.
- Products outside the catalogue: ignore them as targets and in evaluation, but do not discard a whole bin because of them. Label only catalogue products, 0 when absent.
- Test orders: PASS uses an item that is present with its recorded quantity. FAIL uses an absent item or a higher quantity. Mention the limitation that catalogue products almost never share a bin (1,994 of 2,000 bins hold exactly one, 6 hold two).
- Cloud service: my choice, as long as it works.

## Presence threshold
PRESENCE_THRESHOLD = 0.5. A product counts as present when the model output is 0.5 or more. Chosen on the validation set and frozen before the test set was opened. The count is the output rounded, clamped to 1 to 5.

## Final test results (300 test bins)
| Model | Precision | Recall | F1 | Exact count (when present) | MAE |
|---|---|---|---|---|---|
| Baseline (always 0) | 0.000 | 0.000 | 0.000 | 0.000 | 0.088 |
| MobileNetV3-Small | 0.816 | 0.661 | 0.730 | 0.495 | 0.050 |
| ResNet-18 (final) | 0.903 | 0.618 | 0.734 | 0.485 | 0.046 |

## Order test results (600 orders from the 300 test bins, seed 7)
| Model | Order accuracy | False PASS | False FAIL |
|---|---|---|---|
| Baseline | 0.500 | 0 | 300 |
| MobileNetV3-Small | 0.755 | 1 | 146 |
| ResNet-18 (final) | 0.742 | 4 | 151 |
I chose ResNet-18 because its precision is higher, and a wrong PASS costs more than a wrong FAIL. The two models are 8 orders apart out of 600, which I treat as a tie.

## Deployment
Live app: https://smartbinclassifier-praneethguduru.streamlit.app/
I chose Streamlit Community Cloud because it deploys straight from the GitHub repo and is free. The full data is not in the repo, so the app falls back to a small sample/ folder (150 labelled train and validation photos, 100 blind photos). The free host sleeps when idle and wipes the log on restart, so the app has a Download log button.
EOF
tail -5 NOTES.md