"""Experimental revision for frozen R1; original styling and disclosures are retained."""
RTRAIN = RECORDED['residual_training']
RVAL = RECORDED['residual_validation']['records']
NTRAIN = RECORDED['neural_training']['summaries']
NVAL = RECORDED['neural_validation']['records']
FPRIMARY = RECORDED['fusion_primary']['records']
FCONFIRM = RECORDED['fusion_confirmation']['records']
TEST = RECORDED['test']
LEDGER = RECORDED['test_ledger']

page('Selecting Reliable LLM Reasoning: Method and Evidence',True)
p('DASE7506A Advanced Machine Learning Methods | Mini-project 2 | 22 October 2026')
h('1. Research question')
p('When four generated solutions disagree, can a compact verifier improve by combining numeric-answer evidence with checks on the stated conditions and local calculations? We investigate a linear CE baseline, answer pooling, a small evidence residual, candidate attention and a limited fusion grid. Correctness is the final numeric answer accepted by the course scorer; it does not certify every reasoning step.')
p('Our final method, R1, keeps the existing uniform-positive cross-entropy (CE) selector and learns a 50-dimensional evidence residual. The resulting four logits are converted to temperature-one softmax probabilities and completely mean-pooled within equal numeric-answer groups. The new residual improves both training cross-validation partitions, validation and the final fixed test. The neural alternatives and their fixed-weight fusion do not improve this selected method.')
h('Principal result')
table(['Method','Official test','Validation disagreement'],[
 ['CE, without pooling','384/467 (82.23%)','305/355 (85.92%)'],
 ['A: CE + mean pooling','393/467 (84.15%)','308/355 (86.76%)'],
 ['Final R1: A + evidence residual',f"{TEST['correct']}/467 ({TEST['accuracy']:.2f}%)",'312/355 (87.89%)']],[2.25,1.4,1.55])
p('R1 gains four test questions over A, from 393 to 397 out of 467: an improvement of 0.85653 percentage points. On validation, it gains four questions, from 751/798 to 755/798. The final uncompressed inference assets occupy 52,353 bytes, only 1,550 bytes more than A. Inference remains a small CPU/NumPy pipeline without a neural branch or additional candidate generation.')
h('What the comparison can establish')
p('The residual family and regularization are selected on primary training OOF predictions before confirmation and validation. After the neural and fusion studies, R1 is the unique training-ranked eligible candidate and is frozen before its single test. A and CE use saved historical results. Because historical test outcomes and repeated validation results were already known, this is a fixed-bank development study, not an untouched blind test. No test outcome changes the model or its rules.')
p('Sections 2-3 establish the task and baseline. Sections 4-5 study evidence residuals, sections 6-7 test neural interaction and fusion, and sections 8-10 summarize validation, final evaluation and reproduction.',True)

page('2. Evaluation: the difficult groups matter')
table(['Split','Questions','Disagreement','Other groups'],[
 ['Training',4755,2114,2641],['Validation',798,355,443],['Test',467,467,0]],[1.4,1,1.2,1.1])
p('Each question supplies four candidate texts and fixed 1,024-dimensional representations, shaped [N,4,1024]. The output is one candidate index in 0-3. At least one candidate is scorer-correct, and several may be correct. All positive training labels are retained rather than replacing them with a single preferred candidate [1,2].')
p('A disagreement group contains at least two different valid parsed answers. Validation includes 443 other groups, whereas all 467 test questions are disagreement groups. We therefore report disagreement accuracy as the main development measure and complete accuracy alongside it. R1 answers all 443 other validation groups correctly: 755/798 overall (94.61%) becomes 312/355 (87.89%) on disagreement questions.')
h('The scorer defines correctness')
p('The unchanged parser reads after the last #### marker, or otherwise the last simple boxed expression, and canonicalizes its first numeric token as a fraction or decimal. Unparseable candidates receive negative labels. Numeric equivalence, rather than identical answer strings, defines our answer groups. Formatting can therefore affect correctness even when a candidate contains plausible mathematics.')
h('Initial validation references')
table(['Rule','Complete (%)','Disagreement (%)'],[
 ['random_choice','75.56','51.27'],['first_candidate','78.20','54.93'],['majority_vote','86.22','69.01']],[2,1,1.3])
p('The earlier embedding-only logistic model reaches 89.60% complete and 77.18% disagreement validation accuracy; the historical MLP32 reaches 91.85% and 81.69%. These establish that learned candidate scores add information beyond first-choice and voting rules. Comparisons between later objectives use matched features and regularization rather than attributing every difference from this MLP to the loss.')
h('Separate fitting, selection and scoring')
p('Training alone fits weights and normalization statistics. Whole-question folds keep four candidates and any derived objects together. Primary training CV selects residual regularization, the preferred residual family and the fusion weight; confirmation checks that choice. Later validation evaluates fixed candidates. Neither validation nor test is included in a final refit. IDs are used only for alignment, and prediction processes cannot read reference answers.')

page('3. Establishing the CE and pooling baseline')
h('A reusable small linear selector')
table(['Input block','Width','Role'],[
 ['Supplied representation','1,024','Fixed course embedding for each candidate'],
 ['Basic, style and group','30','8 basic, 12 style and 10 within-question features'],
 ['Bounded arithmetic','17','Coverage and consistency; centered within a question'],
 ['Total','1,071','Unchanged inputs for the CE baseline and neural controls']],[1.4,.55,3.15])
p('The CE branch uses training-fold means and standard deviations, replaces scales below 1e-8 with one, and adds no clipping. A shared intercept is fixed to zero. With correctness labels y and m positive candidates, the question loss is the sum of -(y/m) log(p) over the four candidates, plus L2=0.01 on the weights. Questions receive equal weight, including questions with several or four correct candidates.')
p('In our matched linear study, uniform-positive CE exceeds BCE at all eight primary-split regularization settings. After each objective selects its own regularization, disagreement OOF accuracy is 82.45% versus 80.32% in the primary split and 82.73% versus 80.27% in confirmation. Blockwise regularization adds no independent gain. We therefore reuse full-training CE rather than adding another fusion member.')
h('Mean pooling is a decision change')
p('For each valid numeric answer, A assigns all its candidates the mean of their CE softmax probabilities at temperature one. Invalid answers remain independent singletons with their own probabilities. This is a mean, not summed support or majority-vote weighting. Ties depend on answer, text and embedding content, not IDs. Pooling requires no refit.')
table(['Method','Primary OOF / 2,114','Confirmation / 2,114','Validation hard / 355'],[
 ['Unpooled CE',1743,1749,305],['A: complete mean pooling',1745,1750,308]],[1.8,1.4,1.4,1.4])
p('Complete pooling was selected from alpha={0,0.25,0.5,1} using primary OOF before validation. Its historical test result is 393/467, compared with 384/467 for unpooled CE. A becomes the fixed baseline for the new residual, neural and fusion studies; we do not retune pooling in those studies.')
h('Earlier alternatives provide context, not extra final members')
p('Our earlier OOF fusion with soft arithmetic reached 384/467. A set-style comparison residual [3], direct answer-group models and training-only subset augmentation did not displace the CE/pooling baseline. Subsets improved their own group control but remained below CE in both training partitions. These negative results motivate the focused residual study, rather than a larger deployment ensemble.')

page('4. Evidence residual: what additional information helps?')
h('Keep the selector; change the evidence available to a small correction')
p('The existing 17 arithmetic features summarize bounded equality coverage and consistency. A correct-looking calculation can still omit a condition, answer a component instead of a total, or end with a different quantity. We add descriptors of these situations while keeping the official parser, the CE weights and the answer-pooling rule fixed. These are soft features, not a new correctness oracle.')
table(['Family','Extra width','Controlled question'],[
 ['R0: existing arithmetic','17','Does learning a correction from the old arithmetic block suffice?'],
 ['R1: S28 + V22','50','Do condition/target cues and local verification add useful evidence?'],
 ['R2: R1 + relative R20','70','Does another explicit cross-candidate support/conflict block help?']],[1.45,.65,3.1])
h('Condition and target cues: S28')
p('The semantic block describes whether question quantities and salient factors participate in calculations, whether the terminal unit matches the target, and whether intermediate results connect to the final answer. It includes cues for total, remaining, profit and split questions, orphan results, new numerical operands and relation support or conflict. Numeric reuse is descriptive: the same value is not assumed to identify the same real-world entity.')
h('Local verification: V22')
p('The verification block records bounded numeric and simple symbolic checks, check failures and coverage, limited unit consistency, terminal-claim or boxed-answer disagreement with the official parse, and constrained repair hypotheses. Unknown or unsupported checks remain unknown; a failed check does not remove a candidate. R2 adds ranks, dominance and cross-answer support/conflict descriptors beyond the R1 blocks.')
h('Fix extraction boundaries before any fitting')
p('Synthetic examples and label-free TRAIN inspection identify decimal sentence boundaries, detached narrative digits, simple scalar-unit conversions, terminal-unit extraction and mathematical dollar delimiters as failure modes. We fix these before freezing the feature schema. Compound unit expressions abstain rather than forcing a conversion, and candidate text is never executed. Twenty-two synthetic/numerical checks cover these boundaries, empty inputs, permutation and the residual gradient.')
p('The S/V/R definitions are adapted from the <a href="https://github.com/Tianyi-325/DASE7506-MP2-main/blob/8737e0ab5257669e03b14ae6737f0e853c22ab2e/code/math_evidence.py">fixed math_evidence.py source snapshot</a>; the extraction fixes, integration, controlled residual fitting and evaluation are recorded locally. No external checkpoint is run. Our experiments below evaluate our fitted models, not a reproduction of another project.',True)

page('5. Residual training and controlled OOF results')
h('Freeze the CE logits and fit only a small residual')
p('For candidate evidence E and fixed CE logit z, the new score is z + standardize(E)u. Evidence statistics use only the current training fold, with tiny scales set to one and standardized values clipped to [-8,8]. Only u is optimized with the same uniform-positive CE target and equal question weights, plus residual L2. The logits then pass through the unchanged temperature-one softmax and complete mean pooling. A zero residual exactly restores A.')
h('Use fold-specific baselines, not a full-training shortcut')
p('The two five-fold partitions use seeds 20261022 and 20261029, grouped by whitespace-normalized question and stratified by input-defined disagreement. We rebuild the ten fixed CE fold models needed for train-side offsets and reproduce their historical held-out logits and choices exactly. A residual fold uses its own CE fold for both training and held-out rows. Full-training CE is reused only for the final full-training residual, never as a source of OOF predictions.')
p('Each family tests only residual L2={0.03,0.1} on primary OOF. The selected values and preferred family are fixed before confirmation and validation. Selection orders disagreement correct, complete correct and stronger L2. The two partitions use the same training bank and are confirmation splits, not independent datasets.')
residual_rows = [['A: fixed reference','-',1745,1750]]
for family in ['R0','R1','R2']:
    l2 = RTRAIN['selected_l2'][family]
    residual_rows.append([family,str(l2),RTRAIN['primary'][f'{family}_{l2}']['metrics']['hard']['correct'],RTRAIN['confirmation'][family]['metrics']['hard']['correct']])
table(['Method','Residual L2','Primary OOF / 2,114','Confirmation / 2,114'],residual_rows,[1.6,.8,1.5,1.5])
h('New evidence helps; the larger block does not improve selection')
p('R1 repairs 38 primary OOF errors and damages 17, a net gain of 21 over A; confirmation repairs 33 and damages 18, a net gain of 15. Eight of ten folds are nonnegative, so improvement is not uniform across every fold. R0 adds only one primary question and none in confirmation. All 2,641 other training groups remain correct.')
p('R2 does not exceed R1. Their selected L2 values differ, so this is not a pure fixed-regularization causal comparison. At matched L2=0.03, primary R1/R2 counts are 1766/1761; at L2=0.1 they are 1765/1764. The completed comparisons therefore do not support retaining the extra relative block.')
p('The residual study uses 48 fits: 30 primary, 15 confirmation and three full-training fits. All reach the fixed float64 L-BFGS gradient-infinity threshold of 1e-6 within 2,000 iterations. Independent checks reconstruct all 45 held-out predictions and recompute all 48 objectives, gradients and normalization statistics without refitting.',True)

page('6. Neural interaction: a bounded matched comparison')
h('Independent candidate MLP versus a small attention model')
p('We next test whether direct interactions among the supplied representations provide information not captured by the linear method. N0 transforms candidates independently; N1 adds one four-head attention layer without positional encoding. Both use the same original 1,071 inputs, rather than adding the new evidence block to only one model. Existing group statistics remain present in both inputs.')
table(['Shared setting','Value'],[
 ['Projection / feedforward width','64 / 128, ReLU, dropout 0.2'],
 ['Optimizer','AdamW, learning rate 0.001, weight decay 0.03'],
 ['Training budget','30 epochs, batch 128 questions, initialization seed 17'],
 ['Preprocessing','Training-fold standardization; clip [-8,8]; float32'],
 ['Objective','Correct-set probability mass loss + 0.1 candidate BCE'],
 ['Decision','Temperature-one softmax and the same answer mean pooling']],[1.6,3.6])
p('All training questions have equal weight. Each fold and final model resets seed 17; no validation early stopping, epoch search or seed selection is used. Two five-fold partitions plus one full-training model per architecture produce 22 fits. N0 has 85,377 parameters and N1 has 102,145, so the comparison does not hold parameter count exactly constant.')
neural_rows = [['R1: residual reference',1766,1765]]
for model in ['N0','N1']:
    neural_rows.append([model,NTRAIN[f'F20261022_{model}']['vs_R1']['metrics']['hard']['correct'],NTRAIN[f'F20261029_{model}']['vs_R1']['metrics']['hard']['correct']])
table(['Model','Primary OOF / 2,114','Confirmation / 2,114'],neural_rows,[2,1.6,1.6])
h('Attention improves the small neural control, not the final method')
p('N1 gains five and 19 OOF questions over N0, but remains 75 and 77 below R1. Fixed validation gives N0 288/355 and N1 298/355, also below R1 at 312/355. This supports attention within this limited neural recipe, not a general claim that attention beats MLPs or that neural methods cannot beat linear selection.')
p('N1 nevertheless repairs 110 R1 errors in primary OOF and 94 in confirmation, while damaging 185 and 171 successes. This training complementarity qualifies N1 for the preplanned fusion check. The prescribed branch order prefers N1; its validation rank does not choose a different neural branch. All 22 models are exported to NumPy and independently reconstructed, so deployment does not require Torch.',True)

page('7. Limited fusion: complementary errors are not enough')
h('Five fixed weights, without training another model')
p('The fusion score is (1-w)z_R1 + w(t_R1/t_N1)z_N1, followed by the same temperature-one softmax and answer pooling. Each t is the standard deviation of within-question centered logits on the current training fold, floored at 1e-6. Every OOF scale uses only that fold\'s training rows; held-out, validation and test rows do not fit the scale. This matches spread, not calibrated probabilities.')
p('We compare only w={0,0.25,0.5,0.75,1}. Primary disagreement correct, complete correct and smaller neural weight determine the choice. The choice is saved before confirmation is opened. Weight zero strictly returns the original R1 logits and decisions, without an extra temperature or question-wise standardization.')
fusion_rows=[]
for primary,confirmation in zip(FPRIMARY,FCONFIRM):
    w=primary['neural_weight']
    assert w==confirmation['neural_weight']
    ph=primary['metrics']['hard']['correct'];ch=confirmation['metrics']['hard']['correct']
    fusion_rows.append([f'{w:g}',ph,f'{ph-1766:+d}',ch,f'{ch-1765:+d}'])
table(['Neural w','Primary / 2,114','Change','Confirm / 2,114','Change'],fusion_rows,[.8,1.4,.8,1.4,.8])
h('Primary selection and confirmation both retain R1')
p('Weight zero wins on the primary split and remains strongest in confirmation. Even w=0.25 repairs 55 primary errors but damages 59, and repairs 51 confirmation errors but damages 58. Larger weights lose progressively more questions. All weights preserve the 2,641 other training groups and avoid invalid selected answers.')
p('N1 has real rescues, but a single weight cannot identify which questions should trust it. No finer grid, per-question gate or second learned fusion layer is added. This is a negative result for the tested branches, scaling and five weights; it does not rule out every possible fusion strategy.')
h('Do not spend a validation score on an unchanged candidate')
p('Since the selected weight is zero, there is no new nonzero fusion candidate. We retain the original R1 checkpoint, reuse its existing 755/798 and 312/355 validation record, and perform no new official validation scoring or test in this fusion stage. The research alias for w=0 is not added to deployment dependencies.')
p('At w=1, N1 is still multiplied by the prescribed scale ratio. Because softmax probability pooling can depend on logit scale, its 1687/1683 counts differ from raw N1\'s 1691/1688. This is the registered formula, not a changed model. Independent checks reconstruct all ten grid entries and ten fold scales exactly; fusion uses zero new fits.',True)

page('8. Fixed validation and the unique final candidate')
h('Evaluate frozen development candidates against A')
p('The residual study evaluates R0, R1 and R2 once each with the unchanged official predictor and scorer; the neural study evaluates N0 and N1 once each. A uses its saved historical result. No validation labels select epochs, fit normalization or enter the final refit. Fusion selects w=0 on training OOF, so no sixth validation score is needed.')
validation_rows=[['A: pooled CE',751,308,443]]
for model in ['R0','R1','R2','N0','N1']:
    record=(RVAL if model.startswith('R') else NVAL)[model]
    validation_rows.append([model,record['official']['correct'],record['metrics']['hard']['correct'],record['metrics']['other']['correct']])
table(['Method','Complete / 798','Disagreement / 355','Other / 443'],validation_rows,[1.5,1.25,1.65,1.15])
h('R1 improves actual numeric-answer choices')
p('R1 reaches 755/798 (94.61%) overall and 312/355 (87.89%) on disagreement questions. Relative to A, it repairs eight errors and damages four successes: a net gain of four. Thirteen selected numeric answers or parsing statuses change; changes of candidate index within the same answer are not counted as additional improvements. All 443 other validation groups remain correct and no invalid selected answers are introduced.')
p('R2 repairs 12 A errors but damages ten, for a smaller net gain of two. Against R1 it repairs seven and damages nine. N1 repairs 19 R1 errors but damages 33; a label-based oracle could choose the better member, but such an oracle is not an available verifier. The completed TRAIN fusion study, not that diagnostic upper bound, determines deployment.')
h('Use the registered gate and training ranking')
p('The normal gate requires both training OOF disagreement counts to exceed A, validation at least 752/798 overall and 309/355 disagreement, no new invalid choices, and successful implementation, interface and resource checks. R1 and R2 pass. R0 and both neural models do not; there is no distinct fusion candidate.')
p('Eligible candidates are ordered by primary TRAIN disagreement correct, primary complete correct, smaller assets and lower complete prediction cost. Confirmation serves eligibility rather than a second selection rule. R1 ranks above R2, 1766 versus 1764 on primary disagreement, and was already the preferred family before confirmation and validation. Exactly this original R1 checkpoint is frozen for the final test.')
p('The new optimization round totals 80 fits: ten CE fold rebuilds, 48 residual fits and 22 neural fits. It uses five new official validation scores. Prior pooling and group-model studies are historical experiments with separate costs, not hidden members of this final candidate.',True)

page('9. Final test and deployment measurements')
h('One additional frozen test, without retesting A')
p('On 31 October, after candidate selection and deployment checks, we freeze the complete R1 inference code and checkpoint before opening test inputs. A single official prediction process produces all 467 choices. Its prediction file is hashed before an independent scorer can read the test references. No R2, neural or nonzero fusion candidate is tested, and neither A nor CE is rerun.')
table(['Method','Official test','Accuracy'],[
 ['CE, without pooling (historical)','384/467','82.22698%'],
 ['A: mean pooling (historical)','393/467','84.15418%'],
 ['Final R1: evidence residual',f"{TEST['correct']}/467",f"{TEST['accuracy']:.5f}%"]],[2.5,1.2,1.3])
p('R1 improves the retained A baseline by four correct questions, or 0.85653 percentage points. The total difference from unpooled CE is 13 questions, or 2.78373 points, combining the historical pooling gain with the evidence-residual increment. These aggregate differences do not supply an unmeasured R1 test repair/damage breakdown. The test result is retained without changing features, weights, pooling or fusion.')
h('A small standalone CPU inference package')
table(['R1 measurement','Recorded result'],[
 ['Fresh prediction process',f"{LEDGER['prediction_wall_seconds']:.3f} s"],
 ['Independent scoring process',f"{LEDGER['score_wall_seconds']:.3f} s"],
 ['Prediction + scoring',f"{LEDGER['prediction_wall_seconds']+LEDGER['score_wall_seconds']:.3f} s"],
 ['Uncompressed inference assets','52,353 bytes; +1,550 versus A'],
 ['Compute restriction','1 native BLAS thread; 1 logical CPU affinity'],
 ['Runtime dependency','Python 3.12.14, NumPy 2.5.3; no Torch or GPU']],[2.1,3.1])
p('Times include new-process startup, imports, checkpoint loading, preprocessing and separate scoring on the local i7-14700K. Assets are below 16 MiB and local complete time below 30 seconds; the reference Xeon Platinum 8457C runtime remains unmeasured. Windows OS-thread snapshots include auxiliary threads and are not a continuous peak-thread measurement.')
p('Standalone validation inference reproduces all 798 saved R1 choices without reading references or adding a validation score. Empty inputs, 1/3/7/32-question batches, new IDs and candidate permutations pass. Independent checks verify 325 protected/frozen hashes, unchanged original R1 weights and prediction/reference separation. No new fitting or parameter search follows the final test.',True)

page('10. Reproduction, provenance and conclusion')
h('Frozen inference and evidence')
p('The final interface is predict(bank, arrays, checkpoint), returning one original candidate index per question. R1 requires the existing 1,071-input CE feature pipeline and assets, its frozen 50-input evidence residual and normalization, the official parser, and temperature-one complete answer pooling. Inference uses NumPy; it does not fit statistics, load reference answers or generate additional candidates.')
p('The standalone package is R1_portable under V7_Evidence_And_Interaction/runs/stage_d_r1. Its code/submission.py exposes predict and its checkpoint contains the original stage-A R1 assets. PRE_TEST_FREEZE.json records exact source/configuration/weight hashes; TEST_LEDGER.json fixes the prediction hash before scoring. The package and its ZIP require no sibling research code or training cache.',True)
p('Expected official output is 397/467, accuracy 85.01070663811564%. Saved official scores, access logs, full-fold records and independent checks support this result without another test call. The residual L2 is 0.03; the original CE L2 remains 0.01. The neural weight is zero. All selection decisions and unused configurations remain archived.')
p('The evidence comes from one repeatedly studied bank. The two CV partitions are not independent replications, validation has been reused across development, and historical test scores were known before this round. The scorer checks a final parsed number, not all reasoning. The four-question gain is an observed fixed-bank improvement, not a claim of statistical significance or guaranteed transfer.',True)

# Preserve the author-approved disclosure and references byte for byte.
exec(PRESERVED_AI,globals())
h('Conclusion')
p('A compact evidence correction improves the reusable CE-and-pooling selector. R1 adds condition/target and local verification descriptors, improves both training OOF partitions, raises validation from 751 to 755 out of 798, and raises the final test from 393 to 397 out of 467. The larger relative block, small neural models and fixed-weight fusion do not improve the selected method. We select frozen R1 while retaining the limits of repeated development and prior test exposure.')
exec(PRESERVED_REFERENCES,globals())
