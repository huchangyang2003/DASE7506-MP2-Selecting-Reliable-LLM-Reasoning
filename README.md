# Selecting Reliable LLM Reasoning

**Final version: `frozen-v3` (R1). Official test: 397/467, 85.01070663811564%.**

R1 adds a small 50-feature evidence residual to our fixed uniform-positive CE selector, then applies temperature-one softmax and mean probability pooling within equal parsed numeric answers. The residual uses condition/target and local-verification cues. The deployed model is CPU/NumPy only, with no neural ensemble, external weights or new candidate generation.

Read the [10-page report](REPORT.pdf) or [manuscript](REPORT.md) for the experiments. R1 improves the previous A by **4 test questions (+0.85653 percentage points)** and 4 validation questions. Historical test results were already known; this is a fixed-bank development result, not an untouched blind test.

## Run the final checkpoint

Python 3.12 and NumPy 2.5.3. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
$py = '.\.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'

& $py tools/fetch_data.py
# Or use unchanged local course data:
# & $py tools/fetch_data.py --source-dir 'ABSOLUTE_PATH_TO_OFFICIAL_CODE_DATA'
& $py tools/verify_release.py
```

On Linux/macOS, use `python3.12 -m venv .venv`, `.venv/bin/python` and the same thread environment variables. The data helper uses the pinned course source and SHA256 values in [DATA_SOURCES.json](DATA_SOURCES.json). Data is not tracked; only downloading requires internet.

For an optional independent reproduction of the already-frozen result:

```powershell
& $py -B code/predict.py --method code/submission.py --checkpoint checkpoint --data-dir data --split test --output evaluation/repeated-test-predictions.json
& $py -B code/score_predictions.py --data-dir data --split test --predictions evaluation/repeated-test-predictions.json --output evaluation/repeated-test-score.json
```

Expected official fields:

```json
{"protocol":"dase7506-mp2-open-hard-verifier-v5","split":"test","correct":397,"n":467,"accuracy":85.01070663811564}
```

Alternatively, `tools/reproduce_final.py --output-dir evaluation/reproduction_repeat` runs an isolated offline prediction/scoring pair and compares every ID and selected index with the saved result. These commands reproduce a completed evaluation; they are not additional selection experiments. Publication itself adds **no fitting or test scoring**.

## What is retained

| Method | Primary TRAIN hard / 2114 | Confirmation / 2114 | Validation hard / 355 | Test / 467 |
|---|---:|---:|---:|---:|
| Unpooled CE | 1743 | 1749 | 305 | 384 (historical) |
| Previous A: CE + pooling | 1745 | 1750 | 308 | 393 (historical) |
| **Final R1** | **1766** | **1765** | **312** | **397** |

R1 validation is 755/798 overall; all 443 non-disagreement groups remain correct. On validation it repairs 8 A errors and damages 4. R0 adds little; R2, the two small neural models and all nonzero fixed fusion weights do not beat the training-selected R1. Their aggregate results are retained in [evidence/results.json](evidence/results.json), not as additional deployment checkpoints.

The residual family and L2 were chosen on primary TRAIN OOF before confirmation/validation. Both eligible families passed the gate; R1 ranked first by primary TRAIN disagreement correct. Confirmation checked eligibility. Fusion selected neural weight zero, so the original R1 checkpoint was frozen and tested once. No feature, parameter, pooling or fusion change followed that test.

Local fresh prediction plus independent scoring took **2.283 seconds** with one BLAS thread and one logical CPU affinity. Uncompressed inference assets are **52,353 bytes**. These local i7-14700K results do not certify reference-Xeon timing. The final inference files and all four checkpoint files are byte-identical to the version tested in stage D; publication checks their identity without rerunning the test.

## Fixed training recipe

Only the selected recipe is included:

```powershell
& $py -B training/rebuild.py --data-dir data --destination evaluation/training_rebuild
```

This optional command performs **two TRAIN-only fits**: uniform-positive CE on the original 1,071 inputs with L2=0.01, then the 50-dimensional S28/V22 residual with L2=0.03 while holding those CE logits fixed. Both use float64 L-BFGS, at most 2,000 iterations, gradient infinity norm at most 1e-6, and equal question weights. Residual inputs use TRAIN statistics and clipping to [-8,8]; CE inputs are not clipped. The final decision retains alpha=1 and temperature=1. Rebuilt arrays are compared with the shipped arrays, without replacing them. This entry is supplied for reproduction and is not rerun during publication; numerical equality can depend on the platform/BLAS.

The full research round used 80 fits, five new official validation scores and one final R1 test. The two question-grouped five-fold seeds were 20261022 and 20261029. Search implementations, unused model weights, fold caches and verbose logs stay outside the current release. Earlier published implementations remain available through the old tags.

## Repository layout

| Path | Purpose |
|---|---|
| `code/`, `checkpoint/` | The single final R1 inference path and four required asset files |
| `REPORT.pdf`, `REPORT.md`, `report/` | Approved report, manuscript and builder |
| `training/rebuild.py` | Fixed TRAIN-only final recipe; no search or neural training |
| `evaluation/final_test/` | Original R1 predictions, official score, pre-test freeze, call ledger and compact audit |
| `evidence/` | Compact aggregate results and selection provenance |
| `tools/` | Data acquisition, read-only verification and optional isolated reproduction |
| `AI_ASSISTANCE.md`, `ATTRIBUTION.md`, `attribution/` | Unchanged AI disclosure, source attribution and retained course licenses |

[FREEZE.json](FREEZE.json) is a **post-scoring packaging manifest**, not a new pre-test freeze. The authentic pre-test record and prediction-before-scoring ledger are under `evaluation/final_test/`. Historical absolute paths in those original receipts describe the original experiment; runtime uses only this repository and the explicit data directory. The research paths mentioned in the report have the portable equivalents `code/submission.py` and `checkpoint/` here. [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) inventories the current release.

Use the same immutable **[frozen-v3 code tag](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v3)** and **[checkpoint](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v3/checkpoint)**. Previous `frozen-v1`, `frozen-v1.1` and `frozen-v2` are preserved without rewriting their history. Redundant old implementations are removed from current main, not appended to it.

Optional report dependencies are in `report/reporting-requirements.txt`; the original builder uses Arial fonts on Windows. Rebuild without training or scoring:

```powershell
& $py report/build_report.py --output output/REPORT.rebuilt.pdf --markdown output/REPORT.rebuilt.md
```

GitHub publication is separate from any course score-form or issue update.
