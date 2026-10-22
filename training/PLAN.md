# V2 Step 1: controlled training-objective experiment

Declared before any V2 validation results, 22 October 2026 (UTC+8).

The V1 test score has already been seen. This is subsequent development, not a fresh blind test. V1 artifacts remain immutable and V2 contains only supplied training/validation data. No V2 training, selection or scoring process opens public test inputs, references or saved test predictions. Separate V1 preservation checks hash original files, including existing test-related files, solely to confirm immutability; they do not decode or analyze test examples or labels.

## Fixed experiment budget

- Objectives: candidate binary cross-entropy (matched control), positive-vs-negative pairwise logistic ranking, and multi-positive listwise ranking.
- Seeds: 20261022, 20261023, 20261024, shared across all three objectives: **nine fits total**.
- Features: byte-identical copy of the V1 1,054-dimensional group-feature implementation.
- Architecture: the same one-hidden-layer MLP32, ReLU, float32 weights, training-standardized/clipped input.
- Optimizer: Adam, learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, weight L2 0.0001, 60 fixed epochs. No early stopping or hyperparameter expansion after observing results.
- Mini-batch: 128 complete questions (512 candidates). V1 shuffled individual candidates; a NEW BCE control therefore uses the same question batches as the ranking models. Objective effects must be compared to this matched V2 control, not attributed from a raw comparison to the old V1 model.
- Each seed uses exactly the same initialization and question-shuffle schedule across objectives; their hashes are checked.
- Train only: 4,755 questions; validation only: 798, with an input-defined 355-question disagreement subset. No train+validation refit.

## Objectives

Let scores be s_j and P be the set of scorer-correct candidates in a question.

- BCE: mean of binary logistic losses across all four candidates.
- Pairwise: mean softplus(s_negative-s_positive) over all positive-negative pairs in each question, then mean over questions.
- Listwise: logsumexp(scores of all four candidates) - logsumexp(scores of correct candidates).

All positive labels are retained. Questions with four positive candidates have zero data loss/gradient under the ranking objectives, but remain in the question-average denominator. No additional difficulty sampling/weighting, arithmetic features, fusion, new model capacity, answer-group pooling or external data are added in this step. Those are separate potential experiments.

## Evaluation and decision

Report per-seed all-validation and disagreement accuracy from unchanged official prediction/scoring code, inference asset size, external prediction+scoring time, training cost, and paired differences to the matching BCE seed.

Select the candidate objective by mean disagreement accuracy across the three seeds, then mean all-validation accuracy. Seeds are not selected by score and no ensemble is deployed. An objective is described as a promising improvement only if its mean disagreement gain over matched BCE is positive and it improves at least two of the three seeds. Otherwise report insufficient/no improvement. Paired bootstrap intervals (question resampling over seed-averaged correctness differences) are descriptive, not a new independent test or model-selection correction.

Preserve a checkpoint from the predeclared primary seed 20261022 for the selected objective. Do not overwrite the V1 final model or report a new test score.
