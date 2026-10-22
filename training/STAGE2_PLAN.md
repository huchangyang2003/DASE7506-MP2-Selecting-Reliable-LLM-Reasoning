# V2 step 2: disagreement-question loss weighting

Declared before stage-2 validation results, 22 October 2026 (UTC+8).

The stage-1 objective comparison found no clear improvement. This step fixes the BCE objective to isolate weighting rather than combining two changes. V1 and every stage-1 file already present in its manifest remain unchanged. Only supplied train/validation inputs/references are used; no test evaluation or test-based selection is performed. Existing V1 test results have already been seen, so this is later development and not a new blind test. Separate preservation hashing of existing files is integrity checking only.

## Controlled budget: 4 modes x 3 seeds = 12 fits

- Uniform BCE, raw weights 1 for every question.
- `hard_x2`: raw weight 2 for input-defined disagreement questions, 1 for other questions.
- `hard_x4`: raw weight 4 for disagreement questions, 1 for other questions.
- `hard_only`: raw weight 1 for disagreement questions, 0 otherwise. This removes easy-question supervised loss; it does not remove those inputs from standardization or the fixed batch/update schedule.

Disagreement means at least two DIFFERENT valid values under the unchanged official parser; missing answers do not count as an answer group. This definition uses candidate inputs, not reference correctness labels. It identifies 2,114 of the 4,755 training questions. Each mode's raw weights are divided by their full-training-set mean so that the mean weight is one, preserving expected data-loss scale relative to the same regularizer. There is no per-batch renormalization.

All modes retain the same full-training mean/scale, 1,054 V1 features, MLP32/ReLU, float32 weights, Adam settings, 60 epochs, 128-question batches, and seeds 20261022/23/24. Initialization and every epoch's question order must match within a seed. Uniform controls are rerun and their parameters must exactly match the three stored stage-1 BCE models; they are not inferred from a mismatched historical model.

No model capacity, loss family, input feature, additional regularization, fusion, arithmetic check or training epoch is tuned in this step. The 12 fits are fixed before scoring and will not be expanded after observing results.

## Analysis and decision

Use unchanged official validation prediction and scoring. Report full-validation and 355-question disagreement accuracies per seed, seed means/sample standard deviations, the 443 other validation questions, paired comparisons and resource use.

Choose the mode with highest mean disagreement accuracy, then full accuracy. Seeds remain fixed; no best-seed selection or ensemble. A nonuniform mode is a practically promising candidate only if its mean disagreement gain is at least 1 percentage point and it improves at least two of the three matched seeds. Even if this minimum rule is met, a descriptive paired interval spanning zero must be reported as uncertain evidence. Do not replace V1 or report a new test result from this step alone.

Question bootstrap intervals use seed-averaged per-question correctness differences and 5,000 draws, seed 20261022. They are descriptive on reused validation, not an independent generalization evaluation or a correction for seed/method selection.
