# CSC 4501 Project: Neural Network-Based Error Detection

This project builds a neural-network classifier that predicts whether a fixed-length bitstream passes or fails a CRC-8 check.

*I highkey AI generated this readme, but the code is all me*

## What this implements

- Synthetic dataset generation for payload + CRC codewords.
- Controlled bit corruption to create pass/fail examples.
- CRC check baseline (classical deterministic checker).
- PyTorch neural network (MLP) trained on bitstreams.
- Evaluation with accuracy, precision, recall, F1, and confusion counts.
- Smoke test for a quick end-to-end validation.

## Project layout

- `src/index.py` - main experiment script.
- `tests/test_smoke.py` - small test harness.
- `requirements.txt` - Python dependencies.

## Quick start

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Run the default experiment (recommended for grading/ease-of-use):

```bash
python src/index.py
```

This prints:

- selected device (`cuda`, `mps`, or `cpu`)
- training and inference timing
- full metric names (accuracy, precision, recall, f1 score)
- full confusion matrix names (true positives, true negatives, false positives, false negatives)
- direct comparison against the CRC baseline

Important behavior:

- training data is regenerated every run and includes labels (for training)
- if `data/test_data.csv` exists, the script asks whether to reuse it or generate new test data
- if a model checkpoint exists, the script asks whether to use it, continue training it, or start fresh
- CRC pass/fail for test samples is computed internally for evaluation
- optional test export creates a second file with both neural-network prediction and actual CRC outcome per bitstream

The default configuration is intentionally heavier (`samples=20000`, `epochs=75`) to provide high neural-network accuracy for demonstration.

3) Run a custom experiment:

```bash
python src/index.py --samples 5000 --epochs 10 --payload-len 32 --corruption-rate 0.5
```

4) Override default artifact paths:

```bash
python src/index.py \
  --model-checkpoint models/custom_model.pt \
  --test-export data/custom_test_data.csv \
  --test-results-export data/custom_test_data_comparison.csv
```

You can also print sample rows directly in the terminal:

```bash
python src/index.py \
  --samples 200 \
  --show-train-rows 5 \
  --show-test-rows 5
```

5) Run the smoke test:

```bash
python -m pytest tests/test_smoke.py -q
```

## Typical output interpretation

- `CRC baseline metrics` should be near-perfect (often perfect), because labels are defined by CRC pass/fail.
- `Neural Net metrics` show how well the model learns that same decision boundary from raw bits.

## Useful CLI options

- `--samples`: number of generated examples.
- `--payload-len`: payload size in bits (codeword size is `payload + 8`).
- `--corruption-rate`: probability a sample has random bit flips.
- `--epochs`, `--batch-size`, `--lr`, `--hidden`: training controls.
- `--device`: `auto` (default), `cpu`, `mps`, or `cuda`.
- `--model-checkpoint`: checkpoint path to load/save model.
- `--test-export`: optional evaluation export path (default `data/test_data.csv`).
- `--test-results-export`: optional per-bitstream comparison export (default `data/test_data_comparison.csv`).
- `--show-train-rows`, `--show-test-rows`: print sample rows for quick visual checks.

