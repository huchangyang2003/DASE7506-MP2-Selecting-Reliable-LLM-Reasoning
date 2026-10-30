# Attribution and recorded history

The official data, parser, prediction runner and independent scorer come from xudongwu-0/DASE7506-MP2. GSM8K MIT attribution and the supplied Qwen3-0.6B candidate/representation provenance are retained. No additional pretrained weights, external training data or candidate generations are included.

The approved AI assistance statement remains byte-identical in AI_ASSISTANCE.md and the report. The final CE features and solver reuse our earlier project implementation. Mean pooling is our fixed CE-softmax variant, inspired by answer-group/output-pooling ideas in 325200wang/DASE7506-MP2 (4de8e02e28e25048471389f94c3debd9fb8aca50). It does not copy their sigmoid scoring pipeline. Same-answer subset experiments were inspired by CaspianW/dase7506-mp2 (6709b9ff3cfaea1fed0a4f897f6796358ccfcd02); those experiments are not deployed. Broader candidate-comparison research also consulted Devin-Lo1/dase-7506-mp2 (3cbf4d3cd8a6451a88212919d3423795f32b9d7a). No external trained weights or datasets were imported.

frozen-v1.1 retains the earlier OOF/arithmetic model, report and 384/467 result. frozen-v2 contains the newly accepted A, which scores 393/467; CE scores 384/467 as a later supplementary control. Historical test exposure and the post-validation gate exception are disclosed rather than rewritten as a blind test or a passed original gate.

FREEZE.json is a packaging identity record made after scoring; the original pre-test freezes remain under evaluation/. Original source/archive paths and past pending statuses retain their historical meaning. The current final result is in FINAL_RESULT.json. Source adaptations are limited to removing sibling-directory imports, relocating the unchanged CE scorer and making publication wrappers portable; their inference function AST identity and saved test choices are checked.

Only the final A configuration and its CE baseline config/model are deployed. There is no fold or multi-seed ensemble. Checkpoint reproduction and the publication-time fixed training refit are accounted separately from research costs. Reproduction receipts do not claim new method selection or new independent test evidence.
