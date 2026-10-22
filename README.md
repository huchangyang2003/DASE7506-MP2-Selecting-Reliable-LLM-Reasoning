# DASE7506 MP2 — Selecting Reliable LLM Reasoning

**Frozen first release: `frozen-v1`. Official test accuracy: `82.22698072805139` (384/467).**

The submitted method combines an MLP32, an embedding-only logistic scorer and a TRAIN out-of-fold logistic fusion model with soft arithmetic evidence. The primary seed is 20261022. The method and checkpoint are the completed fourth-round solution; none of the proposed supplementary experiments is included.

Read [REPORT.pdf](REPORT.pdf) (**10 pages**) or the editable [REPORT.md](REPORT.md) for the research question, five controlled studies, failures, uncertainty and resource accounting.

## 1. Install

Python **3.12**. Run from this repository's root. PowerShell:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
$py = '.\.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
```

Only NumPy 2.5.3 is needed for training, prediction and scoring. Report generation uses separate optional dependencies. On Linux/macOS use `python3.12 -m venv .venv` and the environment's `python`; the Python script arguments are identical.

## 2. Obtain the unchanged course data

Data is deliberately omitted from this Git repository. The setup helper downloads **only the supplied course files** from official commit `bb801f4f2e7629d270df765ab12c333b2dc616bb` and verifies their SHA-256 digests:

```powershell
& $py tools/fetch_data.py
& $py tools/verify_release.py
```

Alternatively, copy an already-downloaded official `code/data/` directory with `tools/fetch_data.py --source-dir 'ABSOLUTE_PATH_TO_OFFICIAL_CODE_DATA'`. The resulting layout is `data/manifest.json`, `data/gsm8k_train.jsonl`, etc., plus `data/candidates/` for texts and representations. All 12 file hashes are pinned in [DATA_SOURCES.json](DATA_SOURCES.json).

The official package is [xudongwu-0/DASE7506-MP2](https://github.com/xudongwu-0/DASE7506-MP2). Data setup requires internet only when downloading; prediction and scoring run offline. No model weights or external training data are downloaded.

## 3. Reproduce the score without retraining

Use new output names to preserve the original evidence:

```powershell
& $py -B code/predict.py --method code/submission.py --checkpoint checkpoint --data-dir data --split test --output evaluation/repeated-test-predictions.json
& $py -B code/score_predictions.py --data-dir data --split test --predictions evaluation/repeated-test-predictions.json --output evaluation/repeated-test-score.json
```

Expected official fields:

```json
{"protocol":"dase7506-mp2-open-hard-verifier-v5","split":"test","correct":384,"n":467,"accuracy":82.22698072805139}
```

The recorded original outputs are [test-score.json](evaluation/final_test/test-score.json) and [test-predictions.json](evaluation/final_test/test-predictions.json). They are evidence only; the predictor does not load cached predictions or reference answers.

For an isolated offline checkpoint reproduction:

```powershell
& $py -B tools/reproduce_final.py --output-dir evaluation/reproduction_repeat
```

This creates a test-only tree, blocks network/external project reads and denies reference-answer access to the predictor. Only its independent scorer reads gold references. Use another new output directory on later runs.

## 4. Reproduce the fixed training pipeline (optional)

```powershell
& $py -B tools/reproduce_training.py --destination evaluation/training_rebuild
```

This copies only training data into a new working tree, rebuilds the three fixed BCE bases, five-fold OOF models/fusion, frozen arithmetic features and round-4 fusion (**42 fits**). It never reads validation or test, performs no parameter selection and compares rebuilt arrays with the frozen checkpoint. Exact floating-point equality depends on NumPy/BLAS/hardware.

This optional complete retraining was not rerun after the final test result. The completed evidence establishes checkpoint reproduction; the first publication check adds no new training. Original training/search scripts and plans are retained under [training/](training/); historical logs preserve their original path names.

## 5. Result and resource record

| Item | Recorded result |
|---|---|
| Official test | **384/467 = 82.22698072805139%** |
| Historical V1 test | 368/467 = 78.80085653104925% |
| Paired change | 33 repaired, 17 broken, net +16 |
| Final primary-seed validation | 745/798 overall; 302/355 disagreement |
| Full local prediction + independent scoring | 0.8533101 s; fresh processes, startup/loading/preprocessing included |
| Local CPU / parallelism | Intel Core i7-14700K; four BLAS threads; no model worker processes |
| Uncompressed inference assets | 203,273 bytes (198.51 KiB); limit 16 MiB |
| Isolated checkpoint reproduction | All 467 IDs/choices and the score exactly match |

Full measurements: [RESOURCE_REPORT.json](evaluation/final_test/RESOURCE_REPORT.json), [paired comparison](evaluation/final_test/PAIRED_COMPARISON.json), [offline verification](evaluation/reproduction/VERIFICATION.json).

Reference Xeon Platinum 8457C eligibility remains subject to course verification. V1 test accuracy was known before later development; this is not presented as an untouched blind-test study. Method selection used training/validation and the optimized method was frozen before its test evaluation.

## 6. Code/checkpoint identity

Use the same `frozen-v1` tag for the code and checkpoint:

- Code: [frozen-v1 source](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v1)
- Checkpoint: [matching inference assets](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v1/checkpoint)

| Asset | SHA-256 |
|---|---|
| `checkpoint/model.npz` | `deb05856bff0c5938bb833ec5b66b5b4ceba0dc26149c659d299dba46d3f1432` |
| `checkpoint/config.json` | `80778798203066f463aefab58c0db06c9739d9b0cc422e2a06ba2ae6f2e22fa4` |

[FREEZE.json](FREEZE.json) is the immutable pre-test source/asset record. [FINAL_SELECTION.json](FINAL_SELECTION.json) is also historical; its pending status describes that earlier moment. [FINAL_RESULT.json](FINAL_RESULT.json) is the completed result. [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) covers the files in this publication layout.

## 7. Repository map

| Path | Purpose |
|---|---|
| `REPORT.pdf`, `REPORT.md` | Current 10-page report and editable manuscript |
| `code/`, `checkpoint/` | Exact frozen inference sources and all deployment assets |
| `training/code/`, `training/*PLAN.md` | Training/search implementations and predeclared plans |
| `evidence/development/` | Historical five-round summaries, diagnostics, predictions and logs; no old deployment weights |
| `evaluation/` | Original official score, paired/resource records and offline verification |
| `tools/` | Data setup, integrity and reproduction helpers |
| `report/` | Report source, optional dependencies and visual QA |
| `AI_ASSISTANCE.md`, `ATTRIBUTION.md`, `attribution/` | Current disclosure, historical-state explanation and retained licenses |

Virtual environments, downloaded data, regenerable arrays/caches, old report editions and large delivery ZIPs are excluded.

To rebuild the report on Windows with Arial fonts, install `report/reporting-requirements.txt` separately and run:

```powershell
& $py report/build_report.py --output output/REPORT.rebuilt.pdf --markdown output/REPORT.rebuilt.md
```

The report organization was revised after consulting Dylan0528's public report; it did not change the completed experiments or frozen method. This GitHub release is the artifact publication; completing the course score form and its GitHub issue is a separate step.

