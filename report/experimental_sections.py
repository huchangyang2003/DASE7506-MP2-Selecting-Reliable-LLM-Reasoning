"""Updated experimental narrative; original typography and disclosure are reused."""
page('Selecting Reliable LLM Reasoning: Method and Evidence',True)
p('DASE7506A Advanced Machine Learning Methods | Mini-project 2 | 22 October 2026')
h('1. Research question')
p('When four generated solutions disagree, should a verifier select the highest-scoring candidate or first combine evidence for candidates that give the same numeric answer? We investigate this question through controlled loss comparisons, answer pooling, answer-group models and training-time subset augmentation. Correctness is the final numeric answer accepted by the course scorer; it does not certify every reasoning step.')
p('Our final method is a small linear selector trained with uniform-positive cross-entropy (CE), followed by complete mean pooling within numeric-answer groups. It uses the supplied representations and our existing surface, group and arithmetic features. The pooling step changes the final decision without retraining the CE model. More elaborate group training and subset augmentation do not surpass this simpler method in the completed comparisons.')
h('Principal result')
table(['Method','Official test','Validation disagreement'],[
 ['Earlier OOF + arithmetic','384/467 (82.23%)','302/355 (85.07%)'],
 ['CE, without pooling','384/467 (82.23%)','305/355 (85.92%)'],
 ['Final A: CE + mean pooling','393/467 (84.15%)','308/355 (86.76%)']],[2.2,1.4,1.6])
p('Final A gains nine test questions, or 1.93 percentage points, over both the earlier submission and unpooled CE. In the matched CE comparison it repairs 13 errors and breaks four correct choices. Its uncompressed inference assets occupy 50,803 bytes. One fresh prediction process plus independent scoring takes 1.067 seconds locally using four threads; this does not certify the reference-CPU time limit.')
h('What the comparison can establish')
p('Pooling strength was selected using training OOF predictions before validation. A was frozen before its test run; the unpooled CE control was tested afterward to identify the pooling contribution. Historical test scores and repeated validation results were already known. We report this sequence explicitly as a fixed-bank study, not an untouched blind test, and do not tune parameters from the test outcomes.')
p('Sections 2-4 establish the task, earlier experiments and CE baseline. Sections 5-7 report pooling, group training and validation. Sections 8-10 interpret the paired results, final evaluation and reproducibility evidence.',True)

page('2. Evaluation: the difficult groups matter')
table(['Split','Questions','Disagreement','Other groups'],[
 ['Training',4755,2114,2641],['Validation',798,355,443],['Test',467,467,0]],[1.4,1,1.2,1.1])
p('Each question supplies four candidate texts and fixed 1,024-dimensional representations, shaped [N,4,1024]. The output is one candidate index in 0-3. At least one candidate is scorer-correct, and several may be correct. All positive training labels are retained rather than replacing them with a single preferred candidate [1,2].')
p('A disagreement group contains at least two different valid parsed answers. Validation includes 443 other groups, whereas all 467 test questions are disagreement groups. We therefore report disagreement accuracy as the main development measure and complete accuracy alongside it. Final A answers all 443 other validation groups correctly: 751/798 overall (94.11%) becomes 308/355 (86.76%) on disagreement questions.')
h('The scorer defines correctness')
p('The unchanged parser reads after the last #### marker, or otherwise the last simple boxed expression, and canonicalizes its first numeric token as a fraction or decimal. Unparseable candidates receive negative labels. Numeric equivalence, rather than identical answer strings, defines our answer groups. Formatting can therefore affect correctness even when a candidate contains plausible mathematics.')
h('Initial validation references')
table(['Rule','Complete (%)','Disagreement (%)'],[
 ['random_choice','75.56','51.27'],['first_candidate','78.20','54.93'],['majority_vote','86.22','69.01']],[2,1,1.3])
p('The earlier embedding-only logistic model reaches 89.60% complete and 77.18% disagreement validation accuracy; the historical MLP32 reaches 91.85% and 81.69%. These establish that learned candidate scores add information beyond first-choice and voting rules. Comparisons between later objectives use matched features and regularization rather than attributing every difference from this MLP to the loss.')
h('Separate fitting, selection and scoring')
p('Training alone fits weights and normalization statistics. Whole-question folds keep four candidates and any augmented copies together. Training CV selects the pooling strength and group-model structure; later validation evaluates fixed candidates. Neither validation nor test is included in a final refit. IDs are used only for alignment, and prediction processes cannot read reference answers.')

page('3. Earlier experiments and the CE baseline')
p('Our earlier five studies tested objectives, question weighting, learned fusion, arithmetic evidence and a joint comparison head. The useful outcome was an OOF logistic fusion of two scorers with soft arithmetic features, which became the earlier 384/467 submission. The table summarizes the completed development comparisons.')
table(['Controlled study','Observed validation evidence','Decision'],[
 ['MLP objective','BCE 81.13%; pairwise 80.56%; earlier listwise 81.31% mean disagreement','Retain BCE'],
 ['Question weighting','Uniform 81.13%; hard-only 79.81%','No hard-only gain'],
 ['OOF fusion','81.13% MLP control to 83.29% learned fusion','Retain fusion'],
 ['Soft arithmetic','83.29% fusion control to 84.51% full arithmetic','Retain as soft evidence'],
 ['Joint comparison residual','85.07% mean, but gain too small and inconsistent across seeds','Do not deploy residual']],[1.3,2.7,1.15])
p('The means above are across three separately trained seeds, not an inference ensemble. The set residual used shared candidate transformations and pooled context, inspired by Deep Sets [3]. It did not meet its practical stability rule. These earlier results motivated testing a simpler direct linear selector instead of continuing to add fusion members.')
h('Matched linear loss and regularization study')
p('With the same 1,071 inputs and whole-question five-fold CV, uniform-positive CE exceeds linear BCE at all eight distinct matched regularization settings in the primary split. After each objective selects its own uniform regularization, primary disagreement accuracy is 82.45% for CE and 80.32% for BCE; the second split gives 82.73% and 80.27%. The CE choice uses L2=0.01 for both representation and handcrafted blocks. Blockwise regularization itself adds no independent gain: its selected CE setting is the same uniform value.')
h('Further controls did not displace full-training CE')
p('Disagreement-only CE with full-training normalization reaches 306/355 validation, but its two training-CV changes are -1 and +6 questions. Disagreement-only normalization falls to 692/798 overall and introduces 55 other-group errors. A later expanded-feature configuration reaches 304/355 and is not fully converged under its fixed limit; a shallow-tree candidate reaches 292/355. We retain full-training CE as the reusable baseline, rather than presenting these small or unsuccessful changes as improvements.')

page('4. The linear CE selector and training protocol')
h('Fixed candidate inputs')
table(['Block','Width','Construction'],[
 ['Supplied representation','1,024','Frozen course embedding for each candidate'],
 ['Basic, style and group','30','8 basic, 12 style and 10 within-question group features'],
 ['Arithmetic evidence','17','Bounded numeric coverage and consistency features; centered within each question'],
 ['Total','1,071','No new text generation or external training data']],[1.35,.55,3.2])
p('Each feature is standardized using its training-fold candidate mean and standard deviation. Very small scales are set to one. No clipping is added to this CE branch. A linear function produces four logits z for each question. The shared CE intercept is fixed to zero because a common shift cancels in softmax.')
h('Uniform-positive cross-entropy')
p('Let <i>y</i><sub>i</sub> be the correctness label of candidate <i>i</i>, <i>m</i> the number of positive candidates, and <i>p</i><sub>i</sub> its softmax probability from the four logits. The question loss sums -(<i>y</i><sub>i</sub>/<i>m</i>) log(<i>p</i><sub>i</sub>) over the four candidates. We average this loss equally over training questions and add (0.01/2) times the squared weight norm. All correct candidates contribute to the uniform target.')
p('This differs from our earlier listwise loss, which maximized total probability on any positive candidate. It also differs from candidate-wise BCE. A question with four positive candidates still supplies a uniform CE target, whereas positive-negative ranking has no comparison for that question. These objective differences matter when interpreting the earlier negative ranking study.')
h('Whole-question cross-validation')
p('Both five-fold partitions are fixed, stratified by input-defined disagreement and grouped by whitespace-normalized question. Their seeds are 20261022 and 20261029. Standardization and supervised fitting use only the training side of each fold; held-out questions supply OOF scores once. The two partitions reuse the same training bank and are confirmation splits, not independent datasets or deployment ensemble members.')
p('The CE baseline reaches 1,743/2,114 and 1,749/2,114 disagreement OOF correct. Stage 0 checks its saved artifacts, fold coverage, feature schema, gradients and inference interface. A fresh official validation run exactly reproduces 748/798 and 305/355, including all 798 candidate choices. The original CE weights are reused for pooling; the final full-training model does not incorporate validation labels.')
p('Linear fitting uses float64 NumPy L-BFGS with zero initialization, at most 2,000 iterations and gradient infinity norm at most 1e-6. CE is deterministic for fixed inputs and fold assignments; the two CV seeds change folds, not optimizer initialization. Arithmetic checking remains bounded soft evidence, not execution of candidate code or a hard correctness veto.')

page('5. Answer pooling without retraining')
h('From candidate confidence to numeric-answer evidence')
p('For candidate <i>i</i>, let <i>G</i>(<i>i</i>) contain candidates with the same valid parsed numeric answer, and let <i>u</i><sub>i</sub> be their mean CE softmax probability at temperature one. The pooled score is <i>s</i><sub>i</sub> = (1-alpha)<i>p</i><sub>i</sub> + alpha<i>u</i><sub>i</sub>. We return an original candidate with maximum pooled score. Unparseable candidates remain separate singletons and retain their own scores; no additional validity mask is introduced.')
p('At alpha=1, all candidates sharing an answer receive the mean probability for that answer group. This is not a sum over group members or an added majority-vote weight. It tests whether averaging evidence for one answer is preferable to allowing a single high-scoring candidate to determine the outcome. The CE logits, weights and feature pipeline remain unchanged.')
h('Four fixed strengths, one training-selected candidate')
table(['alpha','Primary OOF / 2,114','Confirmation OOF / 2,114'],[
 ['0: unpooled CE',1743,1749],['0.25',1737,1752],['0.50',1744,1763],['1.00: selected A',1745,1750]],[1.6,1.8,1.8])
p('Selection uses primary-split disagreement correct, then complete correct, then smaller alpha. Alpha=1 wins on the primary split, improving CE by two questions. Confirmation improves by one question. Although alpha=0.5 is strongest on the confirmation split, we do not switch after seeing it: that split checks the selected strength rather than selecting a second winner.')
h('Deterministic choices and interface checks')
p('Alpha=0 explicitly uses the original CE argmax and reproduces it candidate by candidate. At nonzero alpha, score ties are resolved from the parsed answer, candidate text and embedding content rather than question IDs or a preferred input position. Choosing another representative of the same answer is not counted as an accuracy gain.')
p('Synthetic and deployment checks cover numeric-equivalent answers, empty inputs, new IDs, different batch sizes, batch independence and candidate reorderings. Reordering is judged by selected numeric answer when equivalent candidates change index. The frozen A checkpoint contains the original CE assets and a small alpha=1, temperature=1 configuration; pooling requires no additional fit or learned parameters.')
p('This experiment isolates a decision change. It does not train a new answer-group representation, resample candidates or recalibrate temperature. Those mechanisms would mix several explanations for a gain, so group learning and subset augmentation are evaluated separately in the next study.')

page('6. Group models and subset augmentation')
h('Matched linear comparisons')
p('B1 uses a candidate-level pairwise logistic model; B2 averages features within each valid numeric-answer group; B3 also supplies answer support and group embedding dispersion. All share 1,024 supplied dimensions plus six bounded numeric features, fixed L2=0.01, no intercept and the same solver. B3 has two additional context features. These models choose among parseable candidates or groups, falling back to index zero when all are unparseable; unlike A they apply validity filtering.')
table(['Model','Primary OOF / 2,114','Confirmation / 2,114'],[
 ['CE reference',1743,1749],['B1: candidate pair',1713,1707],['B2: group mean',1640,1627],
 ['B3: group mean + context',1713,1728],['C: B3 + subset augmentation',1734,1733]],[2,1.6,1.6])
p('Each informative question receives equal total pairwise-loss weight. In B1, inverse answer-group sizes prevent repeated candidates from multiplying one answer-pair contribution. B2/B3 compare valid positive and negative answer groups uniformly. Normalization is fitted on the original training fold, not on held-out or validation rows. B3 is selected over B2 using primary training OOF; B1 remains a matched control.')
h('Same-answer nonempty subsets, training only')
p('C fixes B3 and enumerates every nonempty subset within each answer group. Each subset recomputes mean embedding, support and dispersion while retaining the same answer and label. Original question weight and original answer-pair weight remain fixed; their positive-negative subset combinations divide that weight uniformly. All subsets of an original question remain inside its training fold.')
p('C reuses the corresponding unaugmented B3 mean and scale byte for byte. Its final full-training design contains 12,200 subset-pair comparisons from 3,058 original answer pairs across 2,114 informative questions. These are extra constructions from the same questions, not new independent training data. Inference uses only complete original groups, not subset enumeration.')
h('Useful local gain, insufficient overall result')
p('Subset augmentation improves B3 by 21 and five OOF disagreement questions, yet remains nine and 16 below CE. It also improves B3 by three validation questions later. This supports a limited within-model augmentation benefit, while the group models do not justify replacing CE. B requires 33 fits and C another 11: two five-fold partitions plus full-training fitting per configuration. All 44 meet the fixed 1e-6 gradient criterion.')

page('7. Fixed-candidate validation results')
p('After training selection and checkpoint freezing, we evaluate four new candidates once each with the unchanged official prediction runner and independent scorer. The CE reference reuses its stage-0 validation. B2 is not opened on validation because B3 was the group model selected on primary training CV; we do not use validation to reverse that choice.')
table(['Method','Complete / 798','Disagreement / 355','Other / 443'],[
 ['CE baseline',748,305,443],['A: complete pooling',751,308,443],
 ['B1: candidate pair',742,299,443],['B3: group + context',742,299,443],
 ['C: subset augmentation',745,302,443]],[2.1,1.2,1.5,1.15])
h('Repair and damage relative to CE')
table(['Method','Repairs','Damages','Net correct'],[
 ['A',6,3,'+3'],['B1',15,21,'-6'],['B3',12,18,'-6'],['C',13,16,'-3']],[1.5,1,1,1.2])
p('All methods preserve the 443 other-group correct choices, so their net changes occur on disagreement questions. A is the strongest fixed candidate at 751/798 (94.11%) and 308/355 (86.76%). C repairs six B3 errors and damages three B3 successes, a net gain of three, but still falls below CE. An augmentation improvement over its own control is therefore not equivalent to an improvement over the research baseline.')
h('A missed the original practical gate by one question')
p('The original research rule required both training-CV disagreement counts to exceed CE, validation disagreement at least 309/355, validation complete at least 748/798, and successful convergence, interface and resource checks. A satisfies the other conditions but reaches 308 rather than 309. None of the candidates passes the complete predeclared rule, and that failed-gate decision is preserved.')
p('After reviewing two completed optimization rounds, we explicitly relax this self-imposed research gate for one frozen A test. It is not a course requirement, and we do not rewrite it as having been met. CE is subsequently tested once as an unpooled control, after A performance is known. This sequence limits the inferential claim; neither result triggers new training, strength search or test-driven case rules.')
p('The answer-group round uses 44 real fits and five official validation scores, including the CE reproduction. No optional CE rebuild is needed. The two later test calls are separately authorized additions, not hidden validation calls or extra configuration searches.',True)

page('8. Interpreting gains and failures')
h('Pooling changes a small number of actual answers')
p('On validation, A changes 484 candidate indices relative to CE, but 475 changes stay within the same valid numeric answer. Only nine change the selected numeric answer or parsing status: six repairs and three damages. On test, 208 indices change, with 191 still selecting the same valid answer; the remaining 17 produce 13 repairs and four damages. Counting index changes alone would greatly exaggerate the effect.')
h('Averaging evidence can also remove a correct peak')
p('Within-answer averaging can help when the mean evidence for a correct answer outranks an isolated incorrect peak. It can hurt when a correct high-scoring candidate shares its answer with a weaker candidate. Our observed repairs and damages establish both outcomes, without proving a universal causal explanation for every question. Alpha=0.25 underperforms CE on primary OOF, so partial averaging is not automatically beneficial.')
h('A better training representation did not follow automatically')
p('The plain group mean loses substantially to the candidate control. Adding support and dispersion recovers some accuracy, and subsets further improve B3, but neither reaches CE across both CV partitions. These findings concern the fixed linear pairwise objective, inputs and weighting we tested. They do not establish that all group models or all candidate augmentation strategies fail.')
h('Arithmetic is still soft evidence')
p('The retained CE input contains our earlier bounded arithmetic features. These describe whether local numeric equalities can be checked, how consistent they are and whether they agree with a candidate final answer. They cannot establish that a calculation models the right quantity or includes every cost. In the earlier validation audit, 31 of 97 candidates with a detected inconsistent equation still have scorer-correct final answers. We therefore do not turn arithmetic inconsistency into a hard rejection rule.')
h('Limits of the evidence')
p('There are 4,755 training questions but only 355 disagreement validation questions. Two CV partitions use the same training bank; they are not independent replications. Validation has been used repeatedly across development, and historical test results were already known. CE testing after A is a useful matched control, but not a prospectively registered independent test of a newly selected strength. No unadjusted interval is presented as proof of a significant improvement.')
p('The scorer evaluates the parsed final number, not full reasoning validity. Supplied representations and surface features may exploit generator-specific regularities. We retain failed and superseded experiments and do not analyze test cases to invent repairs. The evidence supports the selected fixed-bank method, with generalization to a new audit bank still uncertain.')

page('9. Final evaluation and resource measurements')
h('Two frozen tests, with different purposes')
p('A is tested once after its configuration and complete checkpoint are frozen. A later CE test supplies the unpooled control, using the exact same CE config and weights as A\'s baseline subdirectory. A predictions are reused without another inference or scoring run. The historical earlier model is also compared from saved predictions, not retrained.')
table(['Method','Official test','Accuracy'],[
 ['Earlier OOF + arithmetic','384/467','82.22698%'],['CE, without pooling','384/467','82.22698%'],
 ['Final A','393/467','84.15418%']],[2.2,1.25,1.35])
table(['Paired direction','Repairs','Damages','Net'],[
 ['CE versus earlier model',23,23,0],['A versus CE',13,4,'+9'],['A versus earlier model',30,21,'+9']],[2.5,1,1,1])
p('The equal CE and earlier-model totals hide different errors. Against CE, A gains 1.92719 percentage points through the pooling decision and its deterministic selection rule. Against the earlier model, the difference includes both the CE replacement and pooling, so that whole-method comparison alone would not isolate pooling. The supplementary CE result supplies the missing control.')
h('Local complete-process measurements')
table(['Measurement','Final A','Unpooled CE'],[
 ['Fresh prediction process','0.861 s','0.746 s'],['Independent scoring process','0.206 s','0.198 s'],
 ['Prediction + scoring','1.067 s','0.944 s'],['Uncompressed inference assets','50,803 bytes','50,441 bytes'],
 ['Actual BLAS threads','4','4']],[2.3,1.4,1.4])
p('Timing includes startup, imports, checkpoint loading, preprocessing and a separate scorer on the local i7-14700K. CPU execution and actual OpenBLAS thread counts are checked. Assets are below 16 MiB and local complete time below 30 seconds; eligibility on the course reference Xeon Platinum 8457C remains unmeasured.')
p('Predictions are hashed before gold scoring; prediction processes cannot open reference answers. Source and checkpoint hashes match before and after each test, and saved choices independently reconcile the official totals. The answer-group round records 44 fits, five validation scores and two later test scores.',True)

page('10. Reproduction, provenance and conclusion')
h('Frozen inference and evidence')
p('The final inference interface is predict(bank, arrays, checkpoint), returning one original candidate index for each question. It requires the fixed CE feature code, CE weights and normalization, the official parser, and the alpha=1, temperature=1 pooling configuration. Inference uses NumPy; it does not fit statistics, load reference answers or generate additional candidates.')
p('The final checkpoint is V6_Answer_Group_Study/runs/pooling/stage_a/selected_checkpoint. Its baseline subdirectory contains the matching CE config.json and model.npz. The recorded method entry is V6_Answer_Group_Study/code/pooling_submission.py, which reuses the unchanged V3 CE implementation. Exact source and asset hashes are in PRE_TEST_FREEZE.json under the saved A final-test run.',True)
p('Expected official output is 393 correct out of 467, accuracy 84.1541755888651%. The supplementary CE output is 384/467. Saved predictions, official scores, access records, timing, complete-fold evidence and independent read-only audits support these figures. Checking those records does not require another test call. Pooling selection needs no fit; the B/C experiment costs are reported separately from prior CE training.')
p('Original pre-test decisions and budgets remain historical snapshots. The later authorization records explain the 309-to-308 gate exception and supplementary CE control. AI/reuse provenance and attribution for adapted design ideas remain with the experiment plans. Neither a higher confirmation score nor a later test result changes the frozen alpha or training recipe.',True)

# The author-approved disclosure is copied verbatim from the previous report source.
exec(PRESERVED_AI,globals())
h('Conclusion')
p('The final improvement is a simple answer-level decision on a reusable CE selector. Uniform-positive CE provides a competitive small baseline; complete within-answer mean pooling adds nine correct test questions in the matched comparison. Direct group learning and subset augmentation yield useful negative and local results but do not replace this baseline. We adopt frozen A while retaining the limits imposed by repeated development, a small validation set and prior test exposure.')
# Preserve the original references and their link targets without alteration.
exec(PRESERVED_REFERENCES,globals())
