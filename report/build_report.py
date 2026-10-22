"""Build the publication report from recorded results; never train or select models."""
from html import escape
import argparse
import json
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
args=argparse.ArgumentParser()
args.add_argument('--output',type=Path,default=ROOT/'output/pdf/DASE7506_MP2_Final_Report.pdf')
args.add_argument('--markdown',type=Path,default=ROOT/'report/REPORT.md')
opts=args.parse_args()
OUT=opts.output;OUT.parent.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
DEV=ROOT/'evidence/development'
if not DEV.exists():DEV=ROOT.parent/'V2_Ranking_Objective/code/runs'
BASE=ROOT/'evidence/baselines/validation/summary.json'
if not BASE.exists():BASE=ROOT.parent/'DASE7506-MP2-main/code/runs/baselines/validation/summary.json'
rounds=[load(DEV/s/'aggregates.json') for s in ('stage1','stage2','stage3','stage4_final','stage5')]
resource=load(ROOT/'evaluation/final_test/RESOURCE_REPORT.json')
repro=load(ROOT/'evaluation/reproduction/VERIFICATION.json')
ledger=load(ROOT/'evidence/SEARCH_COST_LEDGER.json')
baselines=load(BASE)
for name,file in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf'),('ArialItalic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='ArialItalic',boldItalic='ArialBold')
INK=colors.HexColor('#202020');GREY=colors.HexColor('#666666');RULE=colors.HexColor('#B9B9B9')
MARGIN=56.7;WIDTH=A4[0]-2*MARGIN;PAGES=10
styles={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.5,leading=14.2,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.5,leading=11.2,textColor=GREY,spaceAfter=7),
 'title':ParagraphStyle('title',fontName='ArialBold',fontSize=18,leading=23,textColor=INK,spaceAfter=12),
 'head':ParagraphStyle('head',fontName='ArialBold',fontSize=15,leading=20,textColor=INK,spaceAfter=12),
 'sub':ParagraphStyle('sub',fontName='ArialBold',fontSize=11,leading=15,textColor=INK,spaceBefore=8,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=9.0,leading=12.0,textColor=INK),
 'cellhead':ParagraphStyle('cellhead',fontName='ArialBold',fontSize=9.0,leading=12.0,textColor=INK),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=7.8,leading=10.5,textColor=INK,spaceAfter=8),
}
story=[];markdown=[]
def p(text,small=False):
    story.append(Paragraph(text,styles['small' if small else 'body']))
    md=re.sub(r'<a href="([^"]+)">([^<]+)</a>',r'[\2](\1)',text)
    markdown.append(re.sub('<[^>]+>','',md).replace('&lt;','<').replace('&gt;','>')+'\n')
def h(text):
    story.append(Paragraph(text,styles['sub']));markdown.append('### '+text+'\n')
def page(title,first=False):
    if not first:story.append(PageBreak())
    story.append(Paragraph(title,styles['title' if first else 'head']))
    markdown.append(('# ' if first else '## ')+title+'\n')
def table(headers,rows,ratios=None):
    ratios=ratios or [1]*len(headers)
    widths=[WIDTH*r/sum(ratios) for r in ratios]
    data=[[Paragraph(escape(str(v)),styles['cellhead' if i==0 else 'cell']) for v in row] for i,row in enumerate([headers]+rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EFEFEF')),
      ('LINEABOVE',(0,0),(-1,0),.6,INK),('LINEBELOW',(0,0),(-1,0),.5,RULE),('LINEBELOW',(0,-1),(-1,-1),.5,RULE),
      ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,9)])
    markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    markdown.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows);markdown.append('')
def code(text):
    story.append(Preformatted(text,styles['code']));markdown.extend(['```powershell',text,'```',''])
def results(rows,key):
    return [[r[key],f"{r['accuracy_mean']:.2f} +/- {r['accuracy_sample_std']:.2f}",f"{r['disagreement_mean']:.2f} +/- {r['disagreement_sample_std']:.2f}"] for r in rows]

page('Selecting Reliable LLM Reasoning: Method and Evidence',True)
p('DASE7506A Advanced Machine Learning Methods | Mini-project 2 | 22 October 2026')
h('1. Research question')
p('When four solutions disagree, can explicit but incomplete arithmetic evidence improve a learned answer selector? This study tests that question while also asking whether changing the loss, focusing on harder groups or adding a joint comparison network is a better use of computation. Correctness means the final numeric answer accepted by the course scorer; it does not certify every reasoning step.')
p('The central finding is that learned out-of-fold (OOF) fusion provides the clearest development gain, while arithmetic adds a smaller, uncertain improvement. Larger or more targeted models do not improve automatically. The final method therefore keeps two small scorers and a 25-feature logistic fusion model, using arithmetic inconsistencies as soft evidence.')
h('Principal result')
table(['Frozen method','Official test','Validation disagreement'],[
 ['Historical V1 MLP32','368/467 (78.80%)','290/355 (81.69%)'],
 ['Final OOF + arithmetic','384/467 (82.23%)','302/355 (85.07%)']],[2.2,1.4,1.6])
p('The final system gains 16 test questions, or 3.43 percentage points, over V1. Its uncompressed inference assets occupy 203,273 bytes, and complete local prediction plus independent scoring takes 0.853 seconds on an i7-14700K using four threads. These measurements are below the stated limits locally; they do not certify eligibility on the reference CPU.')
h('What the comparison can establish')
p('V1 test performance was known before the five optimization studies. Later fitting used training only, and method selection used validation. The selected primary-seed system was frozen before its final test evaluation; no model, rule or hyperparameter changed afterward. This is a fixed-bank comparison following earlier test exposure, not a wholly untouched blind-test study.')
p('Sections 2-4 define the task and final method. Sections 5-7 explain which changes were retained and rejected. Sections 8-10 connect failure cases, frozen evaluation and the reproducibility record. All numbers below come from the completed experiments; this editorial revision introduces no supplementary experiment.')

page('2. Evaluation: the difficult groups matter')
table(['Split','Questions','Disagreement','Other groups'],[['Training',4755,2114,2641],['Validation',798,355,443],['Test',467,467,0]],[1.4,1,1.2,1.1])
p('Each question supplies four candidate texts and fixed 1,024-dimensional representations, shaped [N,4,1024]. The output is one candidate index in 0-3. At least one candidate is scorer-correct, and several may be correct. Training retains every positive label rather than inventing a unique correct candidate [1,2].')
p('A disagreement group contains at least two different valid parsed answers. Test contains only such groups, while validation includes 443 easier other groups. Disagreement accuracy is therefore the primary development criterion, with complete accuracy reported alongside it. The final model answers all 443 other validation groups correctly: its 93.36% complete accuracy becomes 85.07% on the 355 disagreement questions.')
h('The scorer defines correctness')
p('The unchanged parser reads after the last #### marker, or otherwise the last simple boxed expression, and canonicalizes its first numeric token as a fraction or decimal. Unparseable candidates are negative labels. This makes formatting part of the measured task: a plausible mathematical explanation can still be scored wrong if the parser extracts another number.')
h('Initial validation references')
table(['Rule','Complete (%)','Disagreement (%)'],[[r['method'],f"{r['accuracy']:.2f}",f"{r['disagreement_accuracy']:.2f}"] for r in baselines],[2,1,1.3])
p('An embedding-only logistic scorer reaches 89.60% complete and 77.18% disagreement accuracy; V1 MLP32 reaches 91.85% and 81.69%. V1 shuffles individual candidates, whereas the controlled studies use whole-question batches. Objective changes are consequently judged against a newly matched BCE control, not attributed from a raw comparison with V1.')
p('Training fits all weights and statistics. Validation supports development choices; no train-plus-validation refit occurs. Test references are reserved for frozen scoring, reproduction and the historical paired comparison. IDs serve alignment only, and the predictor receives no reference answers.')

page('3. The submitted selector')
p('The fixed method is mp2-final-oof-arithmetic-seed20261022. It scores each candidate with an MLP and an embedding-only logistic model, constructs agreement and arithmetic evidence, and chooses the maximum fused logit. Exact ties choose the first candidate in the current order.')
table(['Component','Inputs and fixed training settings'],[
 ['MLP32','1,054 inputs: 1,024 embedding dimensions plus 8 basic, 12 style and 10 group features. One ReLU hidden layer; candidate BCE; uniform question weights. Adam lr=0.001, weight L2=0.0001, 60 epochs, batches of 128 whole questions.'],
 ['Embedding logistic','1,024 supplied dimensions; training-candidate mean/std. L2=0.01; deterministic NumPy L-BFGS, gradient tolerance 1e-6, maximum 1,000 iterations.'],
 ['Fusion logistic','25 features with TRAIN OOF mean/std; candidate BCE plus L2/2 times squared weights, L2=0.001. Bias is unpenalized.'],
 ['Deployment seed','20261022, fixed before comparison. Seeds 20261023 and 20261024 measure stability; they are not deployed together.']],[1.1,4])
h('Why two scorers and one fusion model?')
p('The MLP combines representations with surface and group information; the embedding logistic scorer supplies a different, lower-capacity view. The fusion model can learn when to trust either score and how to use answer agreement. Complementarity alone is insufficient: the equal-average control performs worse, which is why the combination is learned rather than assumed useful.')
p('The eight base-fusion features are the two centered logits, centered answer support, two same-answer mean logits, two score-by-support interactions and parsing validity. The remaining 17 features describe arithmetic coverage, consistency and agreement with the candidate\'s own parsed final answer. New arithmetic features are centered within the question.')
p('Deployment uses two models fitted on the full training split and one OOF-fitted fusion model. It contains neither the five fold models nor a three-seed ensemble. No new text is generated, and the method accepts new IDs and different bank sizes without retraining.')
p('The inference entry is code/submission.py; all learned assets are checkpoint/model.npz and checkpoint/config.json. FREEZE.json records their matching source and asset hashes.',True)

page('4. Training fusion without base-model leakage')
h('Five folds, grouped by question')
p('Training fusion on in-sample base predictions could teach it to trust overfitted scores. Instead, five-fold OOF construction ensures every training question receives base predictions from models that did not fit that question. All four candidates remain in one fold. Whitespace-normalized duplicate questions would also stay together; all 4,755 normalized training questions are unique.')
p('The fold seed is 20261022 and the split is stratified by input-defined disagreement status. Within each fold, both standardization and the base models are fitted without the held-out questions. Training labels then supervise the fusion model on these OOF rows. At deployment, the base models are refitted on all training questions; the OOF-trained fusion is retained.')
p('The fusion itself is fitted on the OOF rows, so accuracy on those rows would not be an unbiased cross-validation score of the complete stack. This report instead uses the separate validation split for development comparisons.')
h('What the arithmetic checker measures')
p('The checker re-evaluates bounded numeric equalities with a whitelisted AST and rational arithmetic. It accepts four-function arithmetic, parentheses, small integer powers, numeric percentages and simple expandable LaTeX fractions. Variables, implicit multiplication, unknown syntax and ambiguous trailing percentage units are skipped. Candidate code is never executed.')
p('Strict equality is separate from rounding acceptance, using a fixed tolerance of 0.005 + max(|left|,|right|)/1,000,000. Limits are 20,000 characters and 128 equals signs per candidate, with 256 characters and 100 AST nodes per expression. These rules were frozen before round-4 validation.')
p('Six features describe coverage and format; eleven add correct, rounding-accepted and inconsistent equation counts/ratios, residuals and final-answer self-consistency. Coverage is 84.65% of training candidates and 85.65% of validation candidates. These percentages measure availability, not parser precision or reasoning validity.')
p('Training-input review corrected percentage-fragment extraction and percentage-unit ambiguity before validation. Nine superseded fusion fits were preserved without scoring; only nine final fits were validated. The original implementation failures remain in the audit record. This history is part of the recorded search cost, not an extra successful comparison.')

page('5. Loss and weighting: two negative results')
h('Does ranking supervision improve the same MLP?')
p('Round 1 fixes the 1,054 inputs, MLP32, optimizer, epochs, initialization and question batches within each seed. Only the objective changes. Pairwise loss averages positive-negative softplus comparisons within each question. The listwise loss is logsumexp(all scores) minus logsumexp(correct scores): it rewards total probability on any correct candidate. It is not cross-entropy toward a uniform distribution over positives.')
table(['Objective','Complete mean +/- SD','Disagreement mean +/- SD'],results(rounds[0],'objective'),[1.2,1.6,1.8])
p('The 19,020 training candidates contain 14,353 positives; 2,242 groups have all four positive. Such groups have zero ranking loss but remain in the group-average denominator. Listwise improves disagreement accuracy by only 0.19 points on average, with an interval spanning zero. It meets the initial loose exploratory rule but does not justify replacing BCE. Pairwise is worse.')
h('Does focusing on disagreement help?')
p('Round 2 holds the matched BCE model fixed and changes question weights. Raw weights are divided by their full-training mean, so the overall mean is one; no batch-specific renormalization occurs. The hard-only variant retains the same full-training input standardization and update schedule.')
table(['Weighting','Complete mean +/- SD','Disagreement mean +/- SD'],results(rounds[1],'weight_mode'),[1.2,1.6,1.8])
p('The uniform model exactly reproduces round 1\'s BCE parameters. All 443 other validation groups remain correct, so weighting loses accuracy on the target disagreement subset itself. Uniform BCE is retained. These are negative results under the tested settings, not a general claim that ranking or disagreement weighting cannot work.')
p('Table entries are percentages, with sample SD across three separately trained seeds, not inference ensembles. No additional objectives or weights were searched after observing these results.')

page('6. Fusion and arithmetic: the useful additions')
h('Learning the combination matters')
p('Round 3 compares the matched BCE MLP with equal averaging and eight-feature OOF logistic fusion. The equal average combines sigmoid MLP score, sigmoid embedding score and answer support; learned variants differ only in predeclared L2.')
table(['Round 3 mode','Complete mean (%)','Disagreement mean (%)'],[[r['mode'],f"{r['accuracy_mean']:.2f}",f"{r['disagreement_mean']:.2f}"] for r in rounds[2]],[2,1.3,1.6])
p('L2=0.001 fusion gains 2.16 disagreement points over matched BCE, improving all three seeds; the descriptive paired interval is [0.28,4.04]. Equal averaging declines. The evidence favors learning how the scores interact, rather than treating every scorer as equally trustworthy.')
h('Arithmetic adds information, with uncertainty')
p('Round 4 reuses the same base predictions and L2=0.001. Coverage-only adds six features about checkability; the full variant adds eleven consistency features. This separates some information about equation availability from the consistency bundle.')
table(['Round 4 mode','Complete mean (%)','Disagreement mean (%)'],[[r['mode'],f"{r['accuracy_mean']:.2f}",f"{r['disagreement_mean']:.2f}"] for r in rounds[3]],[2,1.3,1.6])
p('Full arithmetic gains 1.22 disagreement points over round 3, with all seeds improving, and 1.03 points over coverage-only. The respective descriptive intervals are approximately [0.00,2.54] and [-0.19,2.35]. This supports retaining a development candidate, not claiming a confirmed significant gain.')
p('The consistency bundle includes final-answer self-consistency, whose contribution is not isolated. These results therefore do not establish that detecting numerical errors alone caused the gain. The method keeps all arithmetic signals soft, because correct final answers can coexist with flawed calculations.')
p('Across seeds, the retained variant reaches 93.11% +/- 0.66 complete and 84.51% +/- 1.49 disagreement accuracy. The predeclared deployment seed reaches 745/798 and 302/355; the three-seed mean is not its single-model score.')

page('7. A joint comparison head: why it was rejected')
p('Round 5 asks whether a learned interaction among the four candidates improves on round 4. All modes keep its 25 inputs and frozen logit, learning only a neural residual. The set model uses a shared width-16 encoder, then a ReLU layer combining the candidate\'s hidden state with the mean of the other three. Shared transformations make it permutation-equivariant, inspired by Deep Sets [3].')
table(['Residual','New parameters','Complete mean (%)','Disagreement mean (%)'],[[r['mode'],{'control':0,'unary16':705,'unary20':961,'set16':961}[r['mode']],f"{r['accuracy_mean']:.2f}",f"{r['disagreement_mean']:.2f}"] for r in rounds[4]],[1.3,1.1,1.2,1.5])
p('unary16 omits new hidden-state pooling. unary20 matches the set model\'s 961 parameters and depth, although its width differs. All outputs start at zero. unary16/set16 share common initial weights and all neural modes share question batches. BCE, Adam lr=0.001, L2=0.001, 60 epochs and batches of 128 questions are fixed.')
h('The highest mean does not settle the choice')
p('set16 improves mean disagreement accuracy by 0.56 points over round 4, with seed changes of +1.13, +1.41 and -0.85 points. Its descriptive interval [-0.47,1.60] spans zero. It fails the predeclared practical threshold: a gain of at least one point and improvement in at least two seeds. The simpler round-4 method remains selected.')
p('The primary set-model run reaches 749/798 validation, but selecting that favorable run would override the declared stability rule. set16 also exceeds parameter-count-matched unary20 by 1.97 disagreement points, which is evidence about this architectural comparison, not a reliable improvement over round 4.')
h('What this negative result changes')
p('The experiment supports studying candidate interaction, but not deploying the tested residual. It is not rejected for exceeding the local resource limits; the evidence for extra accuracy is too small and inconsistent. No residual strength, extra epoch budget or seed was tuned afterward, and no round-5 variant was evaluated on test.')
p('The final method consequently retains BCE, uniform weights, OOF fusion and soft arithmetic. Ranking objectives, disagreement weighting, equal averaging and the joint residual remain documented experiments rather than hidden components of the submitted system.')

page('8. Failure cases: arithmetic is not understanding')
p('The following validation cases were reviewed before final evaluation. Test is used for frozen aggregate reporting, not for new rules. AI-assisted case review is disclosed and is not independent human verification; complete candidate texts and case lists remain in the experiment evidence.')
table(['Case and change','What can be checked'],[
 ['Snail total: 36 to 35','The rejected candidate writes 3+5+7+9+11=36; the sum is 35. The new choice uses 35: a direct arithmetic cue.'],
 ['Coins: 2853 to 2913','The old candidate misadds 2400+480+33. The checker detects the mismatch; the new candidate writes 2913. The unit conversion itself is not verified.'],
 ['Sweater profit: 308 to 980 (loss)','The wrong candidate computes revenue but omits material costs. The correct candidate contains unit/currency syntax the parser skips. Checkability favors the wrong quantity.'],
 ['Food weight: 7 to 6.5 (loss)','The scorer-correct candidate has incorrect unit conversions and an inconsistent sum but reaches the accepted final answer. The arithmetic signal is misleading.'],
 ['TV price (set study)','Both texts box 1. A #### subheading makes one parse as 1152. This is a scoring-format difference, not evidence of better reasoning.']],[1.5,3.4])
h('Why a hard arithmetic veto is inappropriate')
p('Of 97 validation candidates with a detected inconsistent equation, 31 still have scorer-correct final answers. Conversely, correct local operations can model the wrong question. Arithmetic cannot establish that all costs were included, that a unit conversion is valid or that a quantity answers the question asked.')
p('Coverage-only helps distinguish equation availability from consistency, but format remains a possible shortcut. Embeddings and style features may also reward generator-specific patterns. The set study even reintroduces a 300-versus-296 sum error; extra interaction can override useful evidence.')
p('Generalization remains uncertain: there are 4,755 training questions, only 355 disagreement validation questions and three seed runs, with repeated validation selection and a known historical V1 test score. A four-distinct-answer validation subgroup has only 20 questions. The observed fixed-bank gain does not establish a general-purpose reasoning verifier.')

page('9. Frozen evaluation, uncertainty and resources')
p('Only the selected round-4 system, seed 20261022, is evaluated on test. FREEZE.json was created beforehand; source and asset hashes match before and after evaluation. No alternative optimized architecture or stability seed was tested. The historical V1 model is unchanged.')
table(['Both correct','Final only correct','V1 only correct','Both wrong'],[[351,33,17,66]])
p('The final system answers 384/467 correctly, versus 368/467 for V1. It repairs 33 errors and breaks 17 earlier correct choices: net +16, or +3.43 points. A 5,000-resample paired question bootstrap gives a descriptive gain interval of [0.43,6.42] points. The final accuracy Wilson interval is [78.50%,85.43%]. These intervals are unadjusted for repeated development and do not guarantee performance on an audit bank.')
p('The final validation result is 93.36% overall but 85.07% on disagreement groups. Test contains only disagreement groups, so the overall validation score is not a test forecast. Even its disagreement subset was repeatedly used for selection and remains a small development sample.')
table(['Resource or check','Recorded value'],[
 ['Fresh prediction process',f"{resource['prediction_process_wall_seconds']:.3f} seconds"],
 ['Independent scoring process',f"{resource['scoring_process_wall_seconds']:.3f} seconds"],
 ['Complete local evaluation',f"{resource['total_process_wall_seconds']:.3f} seconds"],
 ['Uncompressed inference assets','203,273 bytes (198.51 KiB); limit 16 MiB'],
 ['Offline checkpoint reproduction',f"384/467; all 467 IDs and choices identical; {repro['total_process_wall_seconds']:.3f} seconds"],
 ['Local hardware and parallelism','Intel Core i7-14700K; CPU; four BLAS threads; no model workers']],[1.9,3])
p('Timing includes startup, imports, loading and preprocessing in one fresh prediction process plus a separate scoring process. A live OpenBLAS query confirms four threads. These local measurements do not certify the 30-second limit on the reference Xeon Platinum 8457C.')
p('The isolated reproduction contains test inputs, scoring references and frozen inference artifacts, with no training or validation files. Network access and reads outside the isolated tree/runtime are blocked. The predictor is denied reference-answer access; only the independent scorer can read gold references.')
p(f"The five studies record 78 successful fits, including nine superseded pre-validation fits, and 57 official validation runs including repeated controls. Recorded successful training-entry time totals {ledger['sum_seconds']:.2f} seconds. This excludes parser audits, unmeasured failed attempts, V1 training and development time; it is not total project effort.")

page('10. Reproduction, provenance and conclusion')
h('Run the frozen checkpoint')
p('The publication repository provides the matching code, checkpoint, official score JSONs and experiment evidence. Its README gives exact Python 3.12 setup and data-download instructions. Inference requires NumPy 2.5.3 only. From the repository root, after setup:')
code("$env:OMP_NUM_THREADS = '4'\n$env:OPENBLAS_NUM_THREADS = '4'\n$env:MKL_NUM_THREADS = '4'\npython code/predict.py --method code/submission.py `\n  --checkpoint checkpoint --data-dir data --split test `\n  --output evaluation/repeated-test-predictions.json\npython code/score_predictions.py --data-dir data --split test `\n  --predictions evaluation/repeated-test-predictions.json `\n  --output evaluation/repeated-test-score.json")
p('Expected accuracy is 82.22698072805139, with 384 correct out of 467. Optional training reproduction rebuilds the fixed TRAIN-only pipeline (42 fits), without validation/test access or parameter reselection. It was not rerun after the final score; isolated checkpoint reproduction was verified. Exact commands, seeds, full hashes and original logs accompany the release.')
p('Checkpoint SHA-256 prefixes: model.npz deb05856bff0c593; config.json 80778798203066f4. FREEZE.json stores the full digests. Historical pre-test snapshots retain their original pending statuses; FINAL_RESULT.json and the release README identify the completed result.',True)
h('AI assistance and reuse')
p('AI assistance was used for code, experiment orchestration, arithmetic parsing, interpretation, case review and this report. I reviewed the relevant code, independently conducted the experiments, verified the reported results, and remain fully responsible for the final implementation and all associated claims. Official parser, data and evaluator are unchanged. Candidate representations were supplied by the course; their Qwen3-0.6B provenance is in the original manifest. No Qwen weights, external training data or new candidate generations were used. GSM8K attribution and MIT license are retained.')
p('This publication revision follows the evidence-led organization of Dylan0528\'s public report [5]. That report was read after this model and its five studies were complete; no method, checkpoint or result was changed by the editorial revision.')
h('Conclusion')
p('The useful improvement is selective combination: OOF fusion learns how to combine small scorers, and bounded arithmetic adds imperfect evidence. The negative studies matter equally: more focused loss weighting or a larger comparison head does not automatically improve selection. The frozen result supports this course-bank method, with the limitations above.')
h('References')
p('[1] <a href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md">Course guide and evaluation rules.</a><br/>[2] <a href="https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md">Course installation and evaluation instructions.</a><br/>[3] Zaheer et al. <a href="https://arxiv.org/abs/1703.06114">Deep Sets.</a> NeurIPS 2017.<br/>[4] <a href="https://github.com/openai/grade-school-math">GSM8K source and retained MIT license.</a><br/>[5] <a href="https://github.com/Dylan0528/DASE7506-Project2/blob/b19e272c687d16f2703ebfe191d6a6f82614adca/REPORT.md">Dylan0528, MP2 Method and Evidence.</a> Presentation reference only.',True)

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Arial',8);canvas.setFillColor(GREY)
    canvas.drawRightString(A4[0]-MARGIN,23,f'{doc.page} / {PAGES}');canvas.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=48,bottomMargin=44,
 title='Selecting Reliable LLM Reasoning: Method and Evidence',author='DASE7506A coursework; AI assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
if len(reader.pages)!=PAGES:raise RuntimeError(f'Expected {PAGES} pages; got {len(reader.pages)}. Repair layout before delivery.')
opts.markdown.parent.mkdir(parents=True,exist_ok=True)
opts.markdown.write_text('\n'.join(markdown)+'\n',encoding='utf-8')
print(json.dumps({'pdf':str(OUT),'pages':len(reader.pages),'test_correct':384,'test_accuracy':82.22698072805139}))
