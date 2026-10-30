# DASE7506 MP2 Selecting Reliable LLM Reasoning

**Final release: `frozen-v2`. Official test: 393/467, accuracy 84.1541755888651%.**

The final method is uniform-positive cross-entropy (CE) linear scoring followed by complete mean pooling over candidates with the same parsed numeric answer. Pooling strength is fixed at 1 and temperature at 1. The CE model uses 1,024 supplied representation dimensions plus 30 surface/group and 17 centered arithmetic features. No extra pretrained weights, external training data or new candidate generation is used.

Read [REPORT.pdf](REPORT.pdf) (10 pages) and [REPORT.md](REPORT.md) for our completed experiments. A matched unpooled CE control scores 384/467. A repairs 13 CE errors and damages four CE successes, net +9. This is a fixed-bank result following earlier test exposure, not an untouched blind test.

## Install

Python 3.12 and NumPy 2.5.3. From this repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
$py = '.\.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
```

On Linux/macOS create the environment with `python3.12 -m venv .venv` and use its Python executable. Prediction and training require only NumPy. Optional report building has separate dependencies in `report/reporting-requirements.txt` and uses Arial fonts on Windows.

## Obtain the unchanged course data

Data is excluded from Git. The helper downloads only course files pinned in [DATA_SOURCES.json](DATA_SOURCES.json), or copies an existing official directory:

```powershell
& $py tools/fetch_data.py
# Alternative, using a local copy:
& $py tools/fetch_data.py --source-dir 'ABSOLUTE_PATH_TO_OFFICIAL_CODE_DATA'
& $py tools/verify_release.py
```

The course source is [xudongwu-0/DASE7506-MP2](https://github.com/xudongwu-0/DASE7506-MP2), pinned at `bb801f4f2e7629d270df765ab12c333b2dc616bb`. All 12 file hashes are verified. Internet is needed only for downloading; prediction, scoring and training reproduction run offline.

## Reproduce the final score without retraining

```powershell
& $py -B code/predict.py --method code/submission.py --checkpoint checkpoint --data-dir data --split test --output evaluation/repeated-test-predictions.json
& $py -B code/score_predictions.py --data-dir data --split test --predictions evaluation/repeated-test-predictions.json --output evaluation/repeated-test-score.json
```

Expected official fields:

```json
{"protocol":"dase7506-mp2-open-hard-verifier-v5","split":"test","correct":393,"n":467,"accuracy":84.1541755888651}
```

For an isolated test-only offline reproduction, with prediction denied gold-reference access:

```powershell
& $py -B tools/reproduce_final.py --output-dir evaluation/reproduction_repeat
```

The package has no dependency on sibling research directories or local absolute paths. [Publication verification](evaluation/publication_verification/CHECKPOINT_REPRODUCTION.json) records the completed packaging reproduction, including exact agreement with all 467 previously saved IDs and candidate choices. This is a reproduction call, not a new optimization/test selection experiment. Use fresh output names when repeating it.

The unpooled control can be reproduced with the same CE weights:

```powershell
& $py -B code/predict.py --method code/ce_core.py --checkpoint checkpoint/baseline --data-dir data --split test --output evaluation/repeated-ce-predictions.json
& $py -B code/score_predictions.py --data-dir data --split test --predictions evaluation/repeated-ce-predictions.json --output evaluation/repeated-ce-score.json
```

Expected control result: 384/467, 82.22698072805139%. It is not another deployed checkpoint.

## Reproduce the fixed training recipe

Rebuild the final CE on TRAIN only, retaining the already-selected pooling configuration (one fit):

```powershell
& $py -B training/rebuild.py --destination evaluation/training_rebuild
```

Also reconstruct both five-fold CE OOF partitions and all four predeclared pooling comparisons (11 fits total):

```powershell
& $py -B training/rebuild.py --destination evaluation/training_rebuild_cv --with-cv
```

The two fold seeds are 20261022 and 20261029. Whole questions and normalized duplicates stay together; standardization is fitted within each training fold. Uniform-positive CE uses L2=0.01 on all 1,071 features, fixed-zero intercept, float64 L-BFGS, maximum 2,000 iterations and gradient infinity norm at most 1e-6. Alpha is selected on primary OOF by disagreement correct, complete correct, then smaller alpha; confirmation does not reselect it. The optional reproduction verifies alpha=1 without validation or test access.

Rebuilt arrays are compared to the original frozen CE checkpoint with absolute tolerance 1e-10; numerical equality can depend on the platform and BLAS. The publication check runs one fixed full-training refit, separately accounted as reproduction; it does not replace the submitted weights. The 11-fit CV rebuild is provided but not rerun during publication. Original CE search and experimental source archives are retained as evidence, not as the primary portable execution entry.

## Results and resources

| Method | Validation overall | Validation disagreement | Official test |
|---|---:|---:|---:|
| Earlier OOF + arithmetic | 745/798 | 302/355 | 384/467 |
| Unpooled CE | 748/798 | 305/355 | 384/467 |
| Final A | 751/798 | 308/355 | 393/467 |

A selected alpha=1 on primary TRAIN OOF (1745/2114 versus CE 1743); confirmation gives 1750 versus 1749. Group pairwise models and subset augmentation did not surpass CE across both partitions. All completed fixed-candidate validation results and negative results remain summarized under `evidence/`.

| Original A measurement | Value |
|---|---:|
| Fresh prediction plus independent scoring | 1.06697 seconds |
| Uncompressed inference assets | 50,803 bytes |
| Local CPU | Intel Core i7-14700K |
| BLAS threads | 4 |
| Reference Xeon eligibility | Not certified by local measurements |

The original answer-group round contains 44 fits, five validation scores and two separately authorized final test scores (A and CE once each). Publication reproduction adds one fixed training fit and one repeated A prediction/scoring pair, recorded separately in `evaluation/publication_verification/`. No parameter, alpha or method changes follow those results.

The original practical validation gate was 309/355. A scored 308; the failed gate is retained, with an explicit exception before the single A test. CE was requested afterward as a supplementary matched control. The final A had already been chosen before that control was run. Test references are used for scoring/reproduction, not training, rule construction or lookup tables.

## Code and checkpoint identity

Use the same immutable `frozen-v2` tag for both [code](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v2) and [checkpoint](https://github.com/huchangyang2003/DASE7506-MP2-Selecting-Reliable-LLM-Reasoning/tree/frozen-v2/checkpoint). Original `frozen-v1` and `frozen-v1.1` tags retain earlier versions; their history is not rewritten.

The three deployed files are `checkpoint/config.json`, `checkpoint/baseline/config.json`, and `checkpoint/baseline/model.npz`. CE arrays and all three configuration/asset files are copied byte for byte from the tested A checkpoint. Only imports are relocated into the standalone `code/` directory; inference function ASTs and all 467 outputs are verified identical.

[FREEZE.json](FREEZE.json) is explicitly a **post-scoring packaging manifest** of the relocated files. It does not claim to be a new pre-test freeze. The actual original pre-test records are preserved in `evaluation/final_test/PRE_TEST_FREEZE.json` and `evaluation/ce_control/PRE_TEST_FREEZE.json`. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) explains copied sources and import-only changes; [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) records current file hashes.

## Repository contents

| Path | Contents |
|---|---|
| `code/`, `checkpoint/` | Standalone final inference and all required assets |
| `training/rebuild.py` | TRAIN-only final refit and optional fixed CV reproduction |
| `REPORT.pdf`, `REPORT.md`, `report/` | Approved 10-page report, manuscript and standalone builder |
| `evaluation/final_test/`, `evaluation/ce_control/` | Original A and CE predictions, official scores, resources/access and audits |
| `evaluation/publication_verification/` | Independent packaging and training-reproduction receipts |
| `evidence/` | Compact reported results, seeds, plans and archived experimental implementations |
| `tools/` | Data acquisition, release verification and offline reproduction |
| `AI_ASSISTANCE.md`, `ATTRIBUTION.md`, `attribution/` | Unchanged approved disclosure, updated provenance and licenses |

No old deployment checkpoints, fold weights, caches, virtual environments, course data or large delivery ZIPs are tracked. Historical source/log paths inside evidence refer to their original locations and are not runtime dependencies. The report records the original A method/checkpoint paths; their portable publication equivalents are `code/submission.py` and `checkpoint/`.

Rebuild the report, without any training or scoring:

```powershell
& $py report/build_report.py --output output/REPORT.rebuilt.pdf --markdown output/REPORT.rebuilt.md
```

GitHub artifact publication is separate from submitting/updating the course result form and its GitHub issue.
