# Smart Bin Classifier

My submission for the Applied AI Engineer Intern assignment at Brainwave Nextgen Technologies.
Author: Guduru Praneeth

## What this project does

In an e-commerce warehouse, someone has to check that a bin really holds the items an order asks
for. Doing it by eye is slow and people make mistakes. I built a small system that does the
check from a photo of the bin.

You build an order (products and quantities), pick a bin photo, and press Validate. A model
looks at the photo and works out how many units of each product it can see. Then plain code
compares those counts with the order and says PASS or FAIL.

The rule is simple. An item passes if the bin holds at least the ordered quantity. An order
passes only when every item passes. When something fails, the app says which item and why:
either the item was not found, or fewer units were found than ordered. Anything in the bin
that is not in the order is ignored.

One thing I was careful about: the check only ever gets the photo and the order. It never reads
metadata.

## Live app

https://smartbinclassifier-praneethguduru.streamlit.app/

The free host puts the app to sleep when nobody uses it, so give it a few seconds to wake up.
The log file on a free host is wiped when the app restarts, so I added a Download log button.

## How to run it

1. `git clone https://github.com/Praneethguduru/SmartBinClassifier`
2. `cd SmartBinClassifier`
3. `pip install -r requirements.txt`
4. `streamlit run app/app.py`

The app does not need the full dataset. If the `data/` folder is missing it falls back to the
small `sample/` folder (more on that below).

To re-run the notebooks you need the dataset. Download it (see the Data section), put
`metadata.db` in `data/`, set the photo folder path in `notebooks/metadata.ipynb`, and run the
notebooks in this order: `metadata.ipynb`, `02_baseline.ipynb`, `03_cnn.ipynb`,
`04_evaluation.ipynb`.

## Data

I used the Amazon Bin Image Dataset (536,434 bin photos, resized to 224x224) from Kaggle, with
its metadata database. The original data comes from Amazon Fulfillment Technologies, and the
licence is CC BY-NC-SA 3.0 US, which means non-commercial use only. I used it only for this
assignment. The full citation is at the bottom.

How I built my working data:

- **2,000 bins.** I kept only bins with 1 to 5 units in total, and where the per-product
  quantities add up to that total. Empty and inconsistent records were dropped.
- **15 products in the catalogue.** I counted how many of my 2,000 bins each product appears
  in and kept every product that reaches at least 100. Exactly 15 do, and the 16th has only 4,
  so the cut is clear. Each catalogue product appears in 119 to 177 bins.
- **Labels.** For every bin I store how many units of each of the 15 products it holds, with 0
  when a product is absent. Products outside the catalogue are ignored, but I did not throw
  those bins away.
- **Split.** A random split with seed 42, so every bin lands in exactly one group:

| Group | Share | Bins | Used for |
|---|---|---|---|
| Train | 70% | 1,400 | Learning |
| Validation | 10% | 200 | Choosing the best epoch and the cut-off |
| Test | 15% | 300 | Final numbers only |
| Blind | 5% | 100 | Live demo, photos only, no labels |

- **Leakage checks.** No bin is in two groups, and I hashed every photo to make sure no
  duplicate image sits in two groups. I found none.

The data preparation code is in `notebooks/metadata.ipynb`.

## Models

I tried three approaches and compared them on the same test set.

| Approach | What it is |
|---|---|
| Baseline | Always says 0 for every product |
| MobileNetV3-Small | Pretrained network with the last layer replaced by 15 outputs |
| **ResNet-18 (my final choice)** | Pretrained network with the last layer replaced by 15 outputs |

Each network outputs one number per product, which is its estimate of the count. A product
counts as present when that number is 0.5 or more. I picked 0.5 using the validation set and
then froze it before I opened the test set. Both networks trained for 15 epochs, and I kept
the best epoch by validation F1.

**Why ResNet-18:** its F1 is basically tied with MobileNet, but its precision is higher. In a
warehouse a wrong PASS (the wrong item ships) costs more than a wrong FAIL (someone looks at the
bin again), so I preferred the model that makes fewer wrong PASS decisions.

## Results

All of these are on the 300 test bins, which were never used for training or tuning.

**Finding the products and counting them**

| Model | Precision | Recall | F1 | Exact count (when present) | MAE |
|---|---|---|---|---|---|
| Baseline (always 0) | 0.000 | 0.000 | 0.000 | 0.000 | 0.088 |
| MobileNetV3-Small | 0.816 | 0.661 | 0.730 | 0.495 | 0.050 |
| **ResNet-18** | **0.903** | 0.618 | **0.734** | 0.485 | **0.046** |

The baseline scores about 93% if you count exact matches over every cell, only because most
products are absent from any given bin. That number is misleading, which is why I judge the
models on precision, recall and F1 instead.

**Order PASS or FAIL**

I built 600 test orders from the 300 test bins (seed 7), following the approach the assessor
agreed to:

- PASS order: one catalogue item that is really in the bin, with its recorded quantity.
- FAIL order: either an item that is not in the bin, or one more unit than the bin holds.

| Model | Order accuracy | False PASS (wrong order accepted) | False FAIL (good order rejected) |
|---|---|---|---|
| Baseline | 0.500 | 0 | 300 |
| MobileNetV3-Small | 0.755 | 1 | 146 |
| **ResNet-18** | 0.742 | 4 | 151 |

For a warehouse, the metric that matters most is the false PASS rate, together with precision.
A wrong PASS ships the wrong item. A wrong FAIL only costs a quick re-check. Both models are
very safe on that front (1 and 4 wrongly accepted orders out of 300). The price is that they
reject about half of the good orders. The two models are 8 orders apart out of 600, which I
treat as a tie.

## Limits, stated honestly

- Tape across the bins, blur and items hidden behind other items put a ceiling on accuracy.
  Some photos also do not match their metadata, so some labels are noisy.
- The model misses a lot of products (recall about 0.62), so about half of the good orders get
  rejected.
- Counting three or more identical units is weak because the data has very few examples of it.
- My 15 products almost never share a bin: 1,994 of the 2,000 bins hold exactly one of them
  and only 6 hold two. So my PASS test orders have a single item, and orders with several
  items only show up as FAIL tests.
- Every bin I picked holds at least one catalogue product, so bins with none of them were not
  tested.
- In normal mode the app picks a bin from train and validation photos so you can see the
  workflow. A PASS there shows the app works, not that the model is accurate. The test set and
  the blind bins are the real checks.

I explain six wrong predictions in the presentation (failure analysis).

## Demo sample

The repo has a small `sample/` folder with 150 labelled train and validation photos (about 10
per product) and the 100 blind photos, with no labels. It is only there so the deployed app
has photos to show. The full data is not in the repo, and no test photos are in the sample.
The script that builds it is `scripts/make_sample.py`.

## Logging

Every time someone presses Validate, a row is written to `logs/validation_log.csv` with the
time, the order, the photo id, the model output, the result and how many seconds it took.

## What is where

- `app/` has the Streamlit app (`app.py`), the validate function (`validator.py`) and the bin
  picker (`bin_index.py`), which is the only place labels are read.
- `notebooks/` has data preparation and EDA, the baseline, the CNN training and the evaluation.
- `models/` has the saved models. `resnet18_best.pt` is the final one.
- `outputs/` has the test predictions and the test orders.
- `sample/` is the small demo sample.
- `scripts/` has the sample builder.
- `NOTES.md` has my working notes.

## Questions I asked and the answers

| What I asked | The answer |
|---|---|
| Can I use every product that appears in at least 100 of my 2,000 bins, with no cap? | Yes. Base it on the frequencies in my own 2,000 bins, and document how I built and checked the catalogue. |
| Can I ignore products outside the catalogue? | Yes, as targets and in evaluation, but do not discard a whole bin because of them. Label only catalogue products, with 0 when absent. |
| My catalogue products almost never share a bin. Is a one-item PASS order and an absent-item or higher-quantity FAIL order fine? | Yes. PASS uses an item that is present with its recorded quantity. FAIL uses an absent item or a higher quantity. Mention the limitation. |
| Which cloud service should I deploy on? | My choice, as long as it works. |

## Assumptions I made

- The deadline is the one in the email: Wednesday, 21 October 2026, 11:59 PM IST.
- A small demo sample (150 labelled photos plus the 100 blind photos) in the public repo is
  fine for the deployed demo. I have not had this confirmed, so I am flagging it here.
- Bin ids keep their leading zeros, because `00001.jpg` and `1.jpg` are different bins.

## How I used AI tools

I used Claude as a chat assistant. It explained the task to me step by step, helped me plan the
work, wrote most of the notebook cells, the app code and the evaluation code, and helped me read
the results. I ran every cell myself and checked the output against what I expected, so I can
explain each step.

Two times the AI was wrong and I caught it:

- An early version of the metadata cell turned every product id into its own column. It ran for
  over 14 minutes and produced a broken CSV. I noticed it was far too slow and that the columns
  made no sense, stopped it, and replaced it with a simple table with one row per product per
  bin.
- The script that builds the demo sample rewrote the label file but never deleted photos from
  the previous run. When I counted the files I got 285 photos for 150 labels. I deleted the
  folder and rebuilt it, and the counts matched.
