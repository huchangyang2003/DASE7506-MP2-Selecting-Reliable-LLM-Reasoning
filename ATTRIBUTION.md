# Attribution and history

Official data, numeric parser and independent evaluation tools are from xudongwu-0/DASE7506-MP2. Questions/references derive from GSM8K; the MIT license and attribution are retained in attribution/. Candidate texts and representations were supplied by the course, with Qwen3-0.6B provenance recorded in the original data manifest. No additional pretrained weights, external training data or candidate generations are included.

FREEZE.json and FINAL_SELECTION.json are immutable pre-test records. Their pending/test_evaluated=false fields describe that earlier state; FINAL_RESULT.json records the completed 384/467 result. Historical development notes under evidence/development may also describe their original pre-test state. They are experiment records, not the current release status.

Only checkpoint/model.npz and checkpoint/config.json are deployed. No fold-model or stability-seed ensemble is required. Original experiment source is under training/code; the two reproduction wrappers change file locations only. No supplementary research method is part of frozen-v1.
