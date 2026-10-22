"""Document the fixed set-comparison study and retain stage 4 under its rule."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'code/runs/stage5'


def read(name):return json.loads((RUN/name).read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def preservation():
    v1=ROOT.parent/'DASE7506-MP2-main';checks={}
    manifests=[('v1',v1,'PROJECT_FILE_MANIFEST.json'),('v2_stage1',ROOT,'FILE_MANIFEST.json'),
               ('v2_stage2',ROOT,'STAGE2_FILE_MANIFEST.json'),('v2_stage3',ROOT,'STAGE3_FILE_MANIFEST.json'),
               ('v2_stage4',ROOT,'STAGE4_FILE_MANIFEST.json')]
    for key,base,name in manifests:
        files=json.loads((base/name).read_text())['files']
        for n,r in files.items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==r['sha256'],n
        checks[f'{key}_original_entries_unchanged']=len(files)
    checks['v1_archives_unchanged']={}
    for line in (v1/'deliverables/SHA256SUMS.txt').read_text().splitlines():
        digest,n=line.split();okay=hashlib.sha256((v1/'deliverables'/n).read_bytes()).hexdigest()==digest
        assert okay;checks['v1_archives_unchanged'][n]=okay
    checks['v2_test_files_present']=[str(p) for p in (ROOT/'code/data').rglob('*test*')]
    assert not checks['v2_test_files_present']
    checks['purpose']='Integrity hashing only. Historical test bytes not decoded/analyzed or used for fitting/selection.'
    save(ROOT/'STAGE5_PRESERVATION_CHECK.json',checks);return checks


def main():
    training,rows,aggregates=read('training_complete.json'),read('summary.json'),read('aggregates.json')
    selection,comparisons=read('selection.json'),read('paired_comparisons.json')
    checks=preservation()
    primary=next(r for r in rows if r['mode']=='set16' and r['seed']==20261022)
    aggregate_table='\n'.join(f"| {r['mode']} | {r['accuracy_mean']:.2f} ± {r['accuracy_sample_std']:.2f}% | "
                             f"{r['disagreement_mean']:.2f} ± {r['disagreement_sample_std']:.2f}% |" for r in aggregates)
    seed_table='\n'.join(f"| {r['mode']} | {r['seed']} | {r['correct']}/798 ({r['accuracy']:.2f}%) | "
                        f"{r['disagreement_correct']}/355 ({r['disagreement_accuracy']:.2f}%) |" for r in rows)
    pair_table='\n'.join(f"| {r['seed']} | {r['corrected']} | {r['broken']} | {r['corrected']-r['broken']:+d} |"
                        for r in comparisons['set16_vs_control']['disagreement']['paired_by_seed'])
    strata_table='\n'.join(f"| {r['distinct_valid_answers']} | {r['n']} | {r['control']:.2f}% | {r['unary16']:.2f}% | {r['unary20']:.2f}% | {r['set16']:.2f}% |"
                          for r in read('strata.json'))
    coverage_table='\n'.join(f"| {r['subset']} | {r['n']} | {r['control']:.2f}% | {r['set16']:.2f}% |" for r in read('coverage_strata.json'))
    cmp=comparisons['set16_vs_control']['disagreement']
    capacity=comparisons['set16_vs_unary20']['disagreement']
    fit_sum=sum(r['training_seconds'] for r in training['new_network_fits'])
    runtime=[r['total_wall_seconds'] for r in rows if r['mode']=='set16']
    decision={'decided_at_utc':datetime.now(timezone.utc).isoformat(),
              'best_average_mode_in_stage5':selection['best_average_mode'],
              'passes_minimum_practical_candidate_rule':False,
              'recommended_for_next_freeze':'stage4_arithmetic',
              'recommended_checkpoint':'runs/stage4_final/arithmetic_seed20261022/checkpoint',
              'experimental_checkpoint':'runs/stage5/set16_seed20261022/checkpoint',
              'reason':'Mean disagreement gain +0.5634pp, below predeclared 1pp threshold; two seed gains, one loss; interval crosses zero.',
              'final_model_frozen_or_tested':False,'original_submission_and_v1_unchanged':True}
    assert selection['best_average_mode']=='set16' and not selection['passes_minimum_practical_candidate_rule']
    save(RUN/'decision.json',decision)
    report=f'''# 第五步结果：联合比较有小幅信号，暂不替换第四步

预设的9个神经残差拟合、3个冻结对照重新封装及12次官方验证已完成。联合模型 set16 的完整验证集三种子平均由第四步 **93.11% 提高到93.36%**，分歧题由 **84.51%提高到85.07%**；平均分歧增益仅 **+0.56个百分点**，两个种子改善、一个种子下降，未达到事前 +1个百分点门槛。

因此，本轮结论是：**联合模型保留为实验候选，当前仍推荐第四步 arithmetic 作为后续冻结的主方案**。`selection.json` 记录本轮均值最高的set16；`decision.json` 记录依据采用门槛保留第四步。两者不是同一概念。没有最终冻结、新测试评分或最终提交，也未覆盖任何旧模型。

## 研究设计与已有组内信息

第四步已经使用票数、同答案池化、题内中心化与算术软特征，不能称为完全独立的候选比较。本轮仅检验在相同25维特征上增加**可学习的跨候选非线性交互**。没有添加新输入、外部数据、额外LLM生成或新的算术规则，也未扩展预设模式/训练参数。

设计参考 [Deep Sets](https://arxiv.org/abs/1703.06114) 的集合排列不变/等变建模思想。本实现为自主设计的mean-other残差网络，未导入论文权重，不是完整论文复现；论文提供结构思路，并不证明本轮方法能提高课程任务准确率。

复用第三步按题五折产生的TRAIN OOF基础分数、第四步已冻结的17个算术输入特征、同种子的基础模型和25维融合标准化。第四步训练得到的logit完全冻结，神经模型只学习它上面的残差。

## 四个固定模式

| 模式 | 结构 | 新增可训练参数 |
|---|---|---:|
| control | 第四步模型原样重新封装 | 0 |
| unary16 | 两层宽16的共享ReLU候选残差网络 | 705 |
| unary20 | 两层宽20的共享ReLU候选残差网络 | 961 |
| set16 | 宽16候选编码 + 其他三个候选隐状态均值 + 第二层融合 | 961 |

set16 的候选编码为 h_i=ReLU(x_i A+a)，其他候选上下文为 c_i=(sum(h)-h_i)/3；第二层 ReLU(h_i B+c_i C+b)，输出标量残差加回冻结logit。相同变换用于全部候选，候选次序重排时相应分数一起重排，没有位置embedding或题号查表。

unary16不增加新的隐状态池化，但它的25维输入仍包含既有组内信息。unary20与set16精确匹配参数数量、层数，但隐藏宽度不同；不能声称所有容量性质都严格相同。unary16/set16公共参数初值逐字节相同，set的额外C零初始化。所有末层向量/偏置零初始化，最初全部分数与第四步逐项一致。

所有残差网络采用相同候选BCE、Adam lr=0.001、权重L2=0.001、60固定epochs、128完整题一批；不改排名目标、不早停、不搜索残差融合比例。残差支路对标准化输入固定裁剪[-8,8]并转float32，冻结logit仍使用原始未裁剪标准化输入。相同种子的三种神经模式共享完全相同的题目批次顺序，哈希检查通过。

## 训练/验证边界

新增拟合仅使用训练标签与TRAIN OOF特征；没有新基础模型拟合。所有网络在 {training['finished_at_utc']} 完成后才开始本轮验证；训练进程实际只打开训练题库，审计拒绝验证/测试文件。OOF原始分数哈希及折题号与第三步记录一致。

冻结的第四步融合器和新增残差都使用训练标签。OOF基础分数仍来自未见过该题的基础模型，但不能把完整两层融合系统在同份TRAIN OOF上的拟合成绩称为无偏交叉验证成绩。每个残差种子沿用对应种子的历史基础模型，因此种子差异同时包括基础模型和残差训练差异。

验证只用于开发模式比较。已知V1历史测试成绩的事实未改变；本轮没有打开测试样本/标签进行训练、选型或评估。独立历史文件完整性哈希仅读取字节，不解码分析测试内容。

## 全部官方验证结果

均值±样本标准差均来自三个独立训练种子的单模型结果，**不是推理集成**。

| 模式 | 完整验证集（798题） | 分歧题（355题） |
|---|---:|---:|
{aggregate_table}

| 模式 | 种子 | 完整正确题数 | 分歧正确题数 |
|---|---:|---:|---:|
{seed_table}

另外443道非分歧题在全部12组中都是443/443。三组control的所有模型数组和798个选择均与第四步同种子完全一致。两个候选残差网络相对第四步均下降，说明仅增加非线性参数并不能保证改善。

## 联合模型的增益与不确定性

set16相对control：完整平均+0.25个百分点，分歧平均+{cmp['mean_gain_percentage_points']:.2f}个百分点。分歧增益逐种子为+1.13、+1.41、-0.85个百分点：

| 种子 | 修正原来错题 | 把原来对题改错 | 净增加正确题 |
|---|---:|---:|---:|
{pair_table}

按三个种子的逐题正确性差异取平均，再进行5,000次题目重采样，分歧增益描述性95%区间为 **[{cmp['question_bootstrap_95_interval'][0]:.2f},{cmp['question_bootstrap_95_interval'][1]:.2f}]个百分点**，跨越零；完整集区间约[-0.21,0.71]。没有校正反复使用验证集和模式选择，不能据此声称统计显著或已确认泛化收益。

与unary16相比，set16分歧均值+1.41个百分点，两种子赢、一种子平，区间约[-0.19,3.19]。与同参数数量的unary20相比，分歧均值+{capacity['mean_gain_percentage_points']:.2f}个百分点，三种子都赢，区间约[{capacity['question_bootstrap_95_interval'][0]:.2f},{capacity['question_bootstrap_95_interval'][1]:.2f}]。这是跨候选结构比本轮候选网络更合适的线索；但对照架构宽度不同，且该优势并未转化为相对第四步足够大的稳定收益，不能由此断言集合模型普遍更优。

预设主种子set16得到 **749/798=93.86%**，分歧 **306/355=86.20%**，相对第四步主种子净+4题。保留它作为实验checkpoint，**不会只凭这个较好的单种子成绩替换第四步**。推荐的第四步主种子仍是745/798=93.36%、302/355=85.07%。请区分单模型成绩与三种子平均成绩。

## 分层和失败案例

| 不同有效答案数 | 题数 | control | unary16 | unary20 | set16 |
|---|---:|---:|---:|---:|---:|
{strata_table}

| 算术覆盖子集 | 题数 | control | set16 |
|---|---:|---:|---:|
{coverage_table}

set16在20道四答案不同题上的均值较高，但样本太少，不能用于追加分组规则。其余子集增益也较小。

主种子有251题的候选索引发生变化，其中只有12题最终正确性变化：8修正、4损失。全部12题的AI辅助案例见 `STAGE5_CASE_REVIEW.md`，包括漏算总量被修正、单位/题意错误被放大，以及因官方解析格式差异而改变评分的例子。文本看似更合理不等于官方解析结果正确；选择改变也不证明模型学会了语义推理。

## 资源与检查

- 本轮完整训练入口耗时 **{training['total_training_wall_seconds']:.2f}秒**，9次神经拟合合计约{fit_sum:.2f}秒；不包含复用的历史基础训练、特征审查、开发和本轮验证成本。
- 主种子set16未压缩推理资产 **{primary['inference_uncompressed_bytes']:,}字节（约{primary['inference_uncompressed_bytes']/1024:.2f}KiB）**，小于16MiB。含全部基础权重、冻结融合参数与联合残差参数，不依赖训练/OOF文件推理。
- set16主种子本机独立官方预测+评分进程合计 **{primary['total_wall_seconds']:.3f}秒**；三个种子范围{min(runtime):.3f}–{max(runtime):.3f}秒。CPU串行、4线程；本地验证计时不替代参考CPU上30秒的正式资格检查。
- 32项检查通过，包括两个结构所有参数的完整有限差分、候选置换的分数和梯度、零输出初始化、上下文能影响其他候选、961参数匹配，以及原27项检查。
- 真实验证集重加载798个选择全部一致，17题新题号一致，候选重排分数相应映射一致，本次原索引及解析答案变化均为0。推理审计只允许读checkpoint两文件，阻止网络；无数据或参考答案读取。
- V1原{checks['v1_original_entries_unchanged']}项、第一步{checks['v2_stage1_original_entries_unchanged']}项、第二步{checks['v2_stage2_original_entries_unchanged']}项、第三步{checks['v2_stage3_original_entries_unchanged']}项、第四步{checks['v2_stage4_original_entries_unchanged']}项和两份原交付包全部未改变。

全部12组结果、训练、配对分析、推理核查及决定记录在 `code/runs/stage5/`；复现见 `STAGE5_REPRODUCE.md`，AI披露见 `STAGE5_AI_DISCLOSURE.md`。

## 后续安排

原先五个优化方向已完成各自预设的一轮研究。建议先收束开发：汇总正负结果，以第四步为主、第五步为补充实验，整理最终方法说明、消融和复现资料，再统一决定最终冻结与评估安排。本轮没有自动开始新的超参数搜索、模型融合比例搜索或测试评估。长期可研究更丰富的候选表示或其他集合架构，但本轮结果不能替它们下结论。
'''
    (ROOT/'STAGE5_RESULTS.md').write_text(report,encoding='utf-8')
    write_cases();write_reproduction();write_overview()
    save(RUN/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'cpu':'Intel Core i7-14700K','thread_limits':4,'parallel_training_workers':1,'new_dependencies':False})
    print(json.dumps({'report':str(ROOT/'STAGE5_RESULTS.md'),'decision':decision,'preservation':checks},indent=2))


def write_cases():
    cases=read('changed_choice_examples.json')
    descriptions={
        22:('修正遗漏总量','第四家店试32双并不是全部商店总数；新选择48正确。只能说明选择变化，不能证明压缩特征网络理解了“总量”。'),
        25:('修正收支方向','原候选把售出收入与购入支出相加，答案20；新候选余额60正确。两段局部算术都可以成立，核心是支出方向。'),
        48:('格式造成的官方解析差异','两段文字都算出并boxed最终答案1。旧候选最后一个####是“New TV”小标题，官方解析器取其后的首个数1152；新候选解析成1。这是格式/解析层面的改善，不能当成推理能力提高。'),
        51:('反例：把书本数当学生数','原9人正确；新候选剩余45本算对，却没有再除以每人5本，答案45错误。'),
        53:('修正游泳时间约束','原候选遗漏领先30秒的约束，得到180秒；新候选150秒正确。候选有变量步骤，但数值模型并不直接求解该变量。'),
        56:('反例：单位关系错误','正确100被换成150，新候选声称1小时=3个半小时。局部3×10和30×5可算对，不能证明单位关系。'),
        86:('反例：覆盖已有算术证据','第四步修正过的小费题又被选成300；56+168+72实际296。软证据可以被残差覆盖，联合结构没有自动保留所有算术改善。'),
        116:('反例：一年月份错误','正确200被换成800；新候选认为一年24个月。随后24×2和25×48-400算式成立但前提错误。'),
        152:('修正年龄时间基准','旧12忽略玛丽四年后的年龄变化，新8正确。不能只凭此例宣称模型学会年龄语义关系。'),
        203:('修正总时长的基数','旧9900将成为大师所需时间只与98小时相乘；新10100把学习规则2小时加入基数，最终正确。'),
        223:('修正漏算小费','旧56只加服务费，漏掉5元小费；新61正确。'),
        248:('修正漏算原有鸽子','旧45仅计算新孵化数量；新65包括原有20只。')}
    lines=['# 第五步逐题案例核查','',
           '范围：预设主种子set16相对第四步control的全部12题正确性变化，包含8次修正和4次损失。AI辅助分析，不是独立人工审核，也未据此重新训练。原候选文本保存在 `code/runs/stage5/changed_choice_examples.json`。','']
    from common import parse_answer
    for index,(title,body) in descriptions.items():
        e=cases[index]
        lines += [f'## {title}','',f"题号：`{e['id']}`；候选 {e['control_index']}→{e['set16_index']}；官方解析答案 {parse_answer(e['control_text'])}→{parse_answer(e['set16_text'])}；正确性 {e['control_correct']}→{e['set16_correct']}。",'',body,'']
    (ROOT/'STAGE5_CASE_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')


def write_reproduction():
    (ROOT/'STAGE5_REPRODUCE.md').write_text('''# 第五步复现

在 `V2_Ranking_Objective/code/` 中运行，依赖仍为原Python3.12.14/NumPy2.5.3，无需联网或新安装。

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$py = '..\\..\\DASE7506-MP2-main\\code\\.venv\\Scripts\\python.exe'
```

仅复现**实验候选**set16主种子的官方验证，不覆盖原输出，期望749/798（93.85964912280701%）：

```powershell
& $py -B run_official.py predict.py --method methods/set_fusion.py --checkpoint runs/stage5/set16_seed20261022/checkpoint --split validation --output runs/stage5/reproduced-validation-predictions.json
& $py -B run_official.py score_predictions.py --split validation --predictions runs/stage5/reproduced-validation-predictions.json --output runs/stage5/reproduced-validation-score.json
```

全训练重现必须使用新目录：

```powershell
& $py -B train_stage5.py --output-dir runs/stage5_reproduction
```

需要第三步既有TRAIN OOF、第四步TRAIN算术特征与三个完整arithmetic checkpoint。训练入口拒绝验证/测试文件。运行生成9个新网络以及3个冻结对照checkpoint；对新目录逐一沿用官方预测/评分命令即可。`evaluate_stage5.py`固定原stage5，已有汇总时拒绝覆盖。

当前推荐主方案仍为第四步，其复现见 `STAGE4_REPRODUCE.md`。不要把本轮实验单种子93.86%与三种子均值93.36%混淆，也不要把任何验证成绩当成新测试成绩。

完成时哈希快照见 `STAGE5_FILE_MANIFEST.json`；新增复现输出不属于原快照，历史产物不得覆盖。
''',encoding='utf-8')
    (ROOT/'STAGE5_AI_DISCLOSURE.md').write_text('''# 第五步AI协助披露

用户授权继续候选联合比较优化。AI助手参考Deep Sets集合建模思想，独立实现小型NumPy残差网络、完整反向梯度和检查，运行预设9个新拟合/12组官方验证，整理全部结果与案例。论文仅作结构参考，未导入外部代码、权重、数据或候选生成。

计划、架构、参数数量、三个种子、训练轮数及采用门槛在本轮验证前固定。没有根据结果扩大搜索。全部新增拟合只使用TRAIN标签及第三步OOF基础分数、第四步冻结算术输入。冻结融合器和残差都使用训练标签，不宣称在同份OOF上得到无偏完整系统准确率。

联合模型相对第四步有小幅均值改善，但未满足事前实际改善门槛，且一个种子下降，因此保留第四步为当前主方案，联合模型作为实验候选。未以较好单种子替代均值/稳定性判定。

案例分析为AI辅助核查，非独立人工评审。包含全部主种子8次修正和4次损失，明确区分最终解析标签、算术证据和完整推理语义。未据此追加规则或重新拟合。

未运行新测试预测/评分、最终冻结或提交；完整性哈希核查仅读历史文件字节，包括旧测试文件，不解码分析其题目/标签。V1、前四步及原交付压缩包全部保持不变。
''',encoding='utf-8')


def write_overview():
    (ROOT/'OPTIMIZATION_REVIEW.md').write_text('''# 五轮优化研究总览

以下全部是同一验证集上的开发结果，不能替代新的独立泛化证据。数值为三个种子的单模型平均，非集成；各行增益均对本轮匹配对照，不应混淆比较基准。

| 研究方向 | 代表配置分歧准确率 | 相对本轮对照 | 当前结论 |
|---|---:|---:|---|
| 1. 训练目标 | listwise 81.31% | +0.19pp | 收益很小、证据不稳，不采用 |
| 2. 分歧加权 | ×2 80.75% | -0.38pp | 本轮加权无收益 |
| 3. OOF互补融合 | stacking 83.29% | +2.16pp | 保留，后续研究基础 |
| 4. 算术软证据 | arithmetic 84.51% | +1.22pp | 当前主方案，仍有不确定性 |
| 5. 候选联合网络 | set16 85.07% | +0.56pp | 未达事前采用门槛，实验保留 |

当前推荐供后续冻结审阅的方案是第四步主种子20261022：完整验证745/798（93.36%），分歧302/355（85.07%）。实验中第五步主种子得到749/798（93.86%），但第三个种子退步，不能只挑这个较好成绩来替换原主方案。

本项目原先V1正式测试368/467（78.80%）保持原样；上述优化均没有新测试分数。继续使用同一验证集意味着选择偏差会累积，不应把多轮小提升简单视为必然可泛化的增益。

原五方向已经完成预设一轮。下一阶段建议收束与整理：汇总方法/消融/失败案例，明确主方案及选择理由，复核可复现性和资产，再统一安排最终冻结和评估。当前没有执行新测试、替换原最终submission或提交外部平台。

详见STAGE1_RESULTS.md、STAGE2_RESULTS.md、STAGE3_RESULTS.md、STAGE4_RESULTS.md、STAGE5_RESULTS.md；第五步采用决定在 `code/runs/stage5/decision.json`。V1及各阶段原文档保留，新增总览不覆盖原报告。
''',encoding='utf-8')


if __name__=='__main__':main()
