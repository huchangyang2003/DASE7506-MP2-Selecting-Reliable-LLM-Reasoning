"""Write the complete arithmetic study, including pre-validation parser revisions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'code/runs/stage4_final'


def read(name): return json.loads((RUN / name).read_text(encoding='utf-8'))
def save(path, value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def preservation():
    v1 = ROOT.parent / 'DASE7506-MP2-main'
    checks = {}
    for key, base, filename in [('v1',v1,'PROJECT_FILE_MANIFEST.json'),('v2_stage1',ROOT,'FILE_MANIFEST.json'),
                                ('v2_stage2',ROOT,'STAGE2_FILE_MANIFEST.json'),('v2_stage3',ROOT,'STAGE3_FILE_MANIFEST.json')]:
        files = json.loads((base / filename).read_text())['files']
        for name, record in files.items():
            assert hashlib.sha256((base/name).read_bytes()).hexdigest()==record['sha256'], name
        checks[f'{key}_original_entries_unchanged'] = len(files)
    archives = {}
    for line in (v1/'deliverables/SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split()
        archives[name]=hashlib.sha256((v1/'deliverables'/name).read_bytes()).hexdigest()==digest
        assert archives[name]
    checks['v1_archives_unchanged']=archives
    checks['v2_test_files_present']=[str(p) for p in (ROOT/'code/data').rglob('*test*')]
    checks['purpose']='Integrity hashing only. Historical test bytes not decoded, analyzed or used for fitting/selection.'
    assert not checks['v2_test_files_present']
    save(ROOT/'STAGE4_PRESERVATION_CHECK.json',checks)
    return checks


def main():
    training, rows, aggregates = read('training_complete.json'), read('summary.json'), read('aggregates.json')
    selection, comparisons = read('selection.json'), read('paired_comparisons.json')
    parser_train=json.loads((ROOT/'code/runs/stage4_parser_v2/summary.json').read_text())
    parser_val=read('parser_validation_summary.json')
    checks=preservation()
    primary=next(r for r in rows if r['mode']=='arithmetic' and r['seed']==20261022)
    old_training=json.loads((ROOT/'code/runs/stage4/training_complete.json').read_text())
    measured_training=old_training['total_training_wall_seconds']+training['total_training_wall_seconds']
    parser_audits=[json.loads((ROOT/f'code/runs/{n}/summary.json').read_text()) for n in
                   ('stage4_parser','stage4_parser_frozen','stage4_parser_v2')]
    measured_parser=sum(r['seconds'] for r in parser_audits)
    agg_table='\n'.join(f"| {r['mode']} | {r['accuracy_mean']:.2f} ± {r['accuracy_sample_std']:.2f}% | "
                        f"{r['disagreement_mean']:.2f} ± {r['disagreement_sample_std']:.2f}% |" for r in aggregates)
    full_table='\n'.join(f"| {r['mode']} | {r['seed']} | {r['correct']}/798 ({r['accuracy']:.2f}%) | "
                         f"{r['disagreement_correct']}/355 ({r['disagreement_accuracy']:.2f}%) |" for r in rows)
    pair_table='\n'.join(f"| {r['seed']} | {r['corrected']} | {r['broken']} | {r['corrected']-r['broken']:+d} |"
                         for r in comparisons['arithmetic_vs_control']['disagreement']['paired_by_seed'])
    strata_table='\n'.join(f"| {r['distinct_valid_answers']} | {r['n']} | {r['control']:.2f}% | {r['arithmetic']:.2f}% |"
                           for r in read('strata.json'))
    coverage_table='\n'.join(f"| {r['subset']} | {r['n']} | {r['control']:.2f}% | {r['arithmetic']:.2f}% |"
                             for r in read('coverage_strata.json'))
    extra=comparisons['arithmetic_vs_coverage']['disagreement']
    chosen_times=[r['total_wall_seconds'] for r in rows if r['mode']=='arithmetic']
    report=f'''# 第四步研究：算术一致性软特征

本轮继续获得正向开发结果。相较第三步选中的 OOF 融合，加入算术软特征后，完整验证集三种子平均准确率由 **92.56% 提高到 93.11%**，355 道分歧题由 **83.29% 提高到 84.51%**，三个种子均改善。达到事前“分歧平均至少 +1 个百分点、至少两个种子改善”的候选门槛；但配对不确定区间仍触及零，不能认定为已确认的泛化改善。

保留预设主种子 20261022：**745/798 = 93.36%**，分歧题 **302/355 = 85.07%**，比第三步同种子净多对 5 题。没有重新评测测试集，没有替换 V1/第三步模型，也没有最终提交。

## 研究设计

复用第三步的同一份训练集五折 OOF 基础预测，以及同一种子的全训练 MLP 与 embedding 线性模型。没有新基础模型拟合。融合器仍用候选 BCE 逻辑回归、固定 L2=0.001、相同三个种子；不选择最好的种子，不跨种子集成。

预设三种模式：

- `control`：第三步原有 8 个融合特征。
- `coverage`：原 8 特征 + 6 个算式覆盖/格式特征（等号数、可检查式数量与比例、跳过数量、无可检查式标记、截断标记）。不包含数字算对/算错信息。
- `arithmetic`：在 coverage 基础上再加 11 个软证据，包括严格相等、容差内相等和不一致的数量/比例、最大/平均相对残差，以及候选自身最终答案与可检查式 RHS 的一致性。共 25 个融合特征。

新增特征在题内中心化；标准化和权重仅拟合训练标签及训练 OOF 基础分数。训练进程拒绝验证/测试文件，全部最终权重于 {training['finished_at_utc']} 拟合完成后才开始本轮验证。三个重拟合 control 的所有参数和验证选择均与第三步逐项完全一致。

算式特征不需要参考答案；“最终答案”是原样官方解析器从候选自身提取的答案。没有修改官方解析、训练标签、数据、预测入口或评分器。没有使用外部数据、模型、生成接口或新增依赖。OOF 预测被用于拟合融合器，不把融合器在同份 OOF 上的拟合成绩冒充完整系统的无偏交叉验证成绩。

## 解析器能检查什么

解析明确写出的数字等式，支持有界四则运算、括号、小整数幂、数字百分比操作数、常见乘除符号和可展开的简单 LaTeX 数值分数。对数字表达式进行 AST 白名单递归，并使用 Fraction 计算；不执行 eval/exec，不运行函数调用，不求解变量或隐式乘法。

严格相等与舍入容差分开统计。固定容差为 `0.005 + max(abs(lhs),abs(rhs))/1,000,000`，没有根据验证成绩调整。每条候选最多前 20,000 个字符、前 128 个等号；表达式最多 256 字符/100 个 AST 节点，绝对值和分母有界。未知表达式跳过，覆盖缺失显式入特征。百分比作为末端 RHS 答案单位存在歧义，例如 `8/25*100=32%`，保守跳过该条而不判错。

不能验证单位转换、题意、步骤完备性、变量关系或一般自然语言论证。容差是启发式，并非完整数值误差传播分析。`bad` 表示本解析器下该条显式等式不一致，不能直接解释成最终答案错误。

## 覆盖与开发修正

最终训练输入检查：19,020 个候选中 **{parser_train['covered_candidates']:,} 个（{parser_train['coverage_percent']:.2f}%）**至少有一条可检查式；87,200 个等号中检查 {parser_train['checked_equations']:,} 条，严格相等 {parser_train['exact_equations']:,} 条，容差接受 {parser_train['accepted_equations']:,} 条，不一致 {parser_train['bad_equations']} 条，涉及 {parser_train['candidates_with_bad_equation']} 个候选。候选覆盖率不等于等式覆盖率，更不是解析精确率。

最终验证输入检查：3,192 个候选中 **{parser_val['covered_candidates']:,} 个（{parser_val['coverage_percent']:.2f}%）**至少有一条可检查式；检查 {parser_val['checked_equations']:,} 条，其中 {parser_val['bad_equations']} 条不一致，涉及 {parser_val['candidates_with_bad_equation']} 个候选。这 97 个候选中仍有 **31 个（31.96%）被官方标签判为最终答案正确**。该关联统计再次说明不能采用“发现错式就淘汰”的规则；无不一致的候选也不能视为推理正确。

在读取本轮验证集之前，训练样本核查修正了两类边界：文字百分比中的数字片段误取，以及百分号作为答案单位的歧义。原始审查及中间版本均保留；中间一次 prefix 变量使用位置错误被单元检查发现，失败日志保留，修复后最终 27 项检查通过。

百分比 RHS 修正前已有 9 次融合拟合完成，这些模型没有评分、没有参与选型。修正后在新目录重新完成预设 9 次融合拟合。因此本轮实际 **18 次融合拟合（9 次被替代 + 9 次最终），仍只运行 9 次官方验证**。这是训练输入审查驱动的实现修复，未根据验证表现扩展模式、参数或规则；最终解析器和预算均在本轮验证前冻结。

训练示例由 AI 助手辅助核查，属于实现检查，并非独立人工审查或随机抽样的解析精度估计。

## 全部结果

下表为三个随机种子的平均值 ± 样本标准差，不是模型集成的准确率。

| 模式 | 完整验证集 | 分歧题 |
|---|---:|---:|
{agg_table}

| 模式 | 种子 | 完整正确题数 | 分歧正确题数 |
|---|---:|---:|---:|
{full_table}

另外 443 道非分歧题在全部九组结果中都是 443/443。相比 control，coverage 的分歧平均仅 +0.19 个百分点；arithmetic 为 **+1.22 个百分点**。相比 coverage，arithmetic 额外 +{extra['mean_gain_percentage_points']:.2f} 个百分点且三个种子均改善，说明本轮增益不能仅由这六个覆盖/格式特征解释。但完整方案还包含最终答案自一致性，当前没有单独隔离“纯等式检查”和“最终答案自一致性”的贡献，不能将所有增益单独归因于计算错误检测。

## 配对变化与不确定性

| 种子 | 修正原来错题 | 把原来对题改错 | 净增加正确题 |
|---|---:|---:|---:|
{pair_table}

算术方案相对 control 的分歧题描述性 95% 配对 bootstrap 区间约 **[0.00, 2.54] 个百分点**（浮点计算下端约 -9e-17，可视为零）；相对 coverage 的区间为 **[-0.19, 2.35]**。完整集相对 control 平均 +0.54 个百分点，区间约 [0.00,1.13]。区间由三个种子逐题正确性差异平均后进行 5,000 次题目重采样得到。

这些区间触及或跨越零，且没有校正重复使用验证集和选型，结论应为“值得保留的正向开发候选”，不应声称统计显著、已稳定泛化或新的测试成绩。历史测试 78.8% 未被本轮重新测量。

## 分层与逐题观察

| 不同有效答案数 | 题数 | 第三步 control 三种子平均 | arithmetic 三种子平均 |
|---|---:|---:|---:|
{strata_table}

有 2 种答案的题平均改善；3 种答案的题整体持平；4 种答案的 20 题由 58.33% 提高到 68.33%，但样本很小，不能作为稳定优势或额外调参依据。

| 可检查式覆盖子集 | 题数 | control 三种子平均 | arithmetic 三种子平均 |
|---|---:|---:|---:|
{coverage_table}

覆盖缺失仍是限制。逐题 AI 辅助案例分析见 `STAGE4_CASE_REVIEW.md`，既保留修正加法/乘法错误的例子，也保留被错式误导、缺少单位/题意理解的反例。模型改变选择还受到原模型分数和所有融合权重共同影响，不能仅由某条特征解释为因果证明。

## 成本和推理检查

- 两个训练入口共计 **{measured_training:.2f} 秒**，含 18 次小型融合拟合；复用的基础训练成本不在该数字内。三次完整训练输入解析审查合计约 **{measured_parser:.2f} 秒**，另有一次编码失败的审查未单独计时；不把这些成功计时冒充全部开发成本。
- 主种子 checkpoint 未压缩 **{primary['inference_uncompressed_bytes']:,} 字节（约 {primary['inference_uncompressed_bytes']/1024:.2f} KiB）**，含两个基础模型及全部融合参数，小于 16 MiB。推理不需要 OOF 或训练数据文件。
- 主种子本机独立官方验证预测+评分进程合计 **{primary['total_wall_seconds']:.3f} 秒**；三个 arithmetic 种子范围 {min(chosen_times):.3f}–{max(chosen_times):.3f} 秒。CPU 串行、4 线程；正式参考 CPU 的 30 秒资格仍未重新认证。
- 27 项单元检查通过，包含算术精确值、舍入、LaTeX、百分比歧义、变量碎片排除、表达式执行边界、候选置换和既有优化检查。
- 最终模型重加载的全部 798 个选择与官方输出相同；17 题更换题号也一致；候选重排的分数相应映射一致，本次选择原索引与解析答案变化均为 0。
- 严格推理审计只允许打开 model.npz/config.json，并阻止网络访问；实际检查期间没有读取数据或参考答案。
- V1 的 {checks['v1_original_entries_unchanged']} 项、第一步 {checks['v2_stage1_original_entries_unchanged']} 项、第二步 {checks['v2_stage2_original_entries_unchanged']} 项、第三步 {checks['v2_stage3_original_entries_unchanged']} 项及原两份交付压缩包全部未改变。完整性核查读取历史文件字节计算哈希，包括旧测试相关文件，但不解码、分析其题目/标签。

选中开发 checkpoint：`code/{selection['development_checkpoint']}`。全部最终九组结果在 `code/runs/stage4_final/`；被替代的九次拟合在 `code/runs/stage4/`。复现见 `STAGE4_REPRODUCE.md`，AI 协助披露见 `STAGE4_AI_DISCLOSURE.md`。

## 下一步

按原路线，下一步可研究四个候选的联合比较（集合模型），再判断是否补充当前共享候选打分器的不足。本轮尚未开展。仍应先完成预设开发比较并统一冻结方案，避免反复通过测试分数驱动修改。
'''
    (ROOT/'STAGE4_RESULTS.md').write_text(report,encoding='utf-8')
    write_cases()
    (ROOT/'STAGE4_REPRODUCE.md').write_text('''# 第四步复现

在 `V2_Ranking_Objective/code/` 中操作，使用既有虚拟环境，无需新增依赖或网络。最终结果目录是 `runs/stage4_final/`，`runs/stage4/` 是验证前修正替代的旧拟合，不能混用。

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
$py = '..\\..\\DASE7506-MP2-main\\code\\.venv\\Scripts\\python.exe'
```

只复现主模型的官方验证，输出到新文件，期望 745/798（93.35839598997494%）：

```powershell
& $py -B run_official.py predict.py --method methods/arithmetic_fusion.py --checkpoint runs/stage4_final/arithmetic_seed20261022/checkpoint --split validation --output runs/stage4_final/reproduced-validation-predictions.json
& $py -B run_official.py score_predictions.py --split validation --predictions runs/stage4_final/reproduced-validation-predictions.json --output runs/stage4_final/reproduced-validation-score.json
```

完整重现特征与训练必须用新目录，历史文件保留：

```powershell
& $py -B audit_arithmetic_train.py --output-dir runs/stage4_parser_reproduction
& $py -B train_stage4.py --output-dir runs/stage4_reproduction --parser-audit-dir runs/stage4_parser_reproduction
```

训练读取第三步已冻结的 TRAIN OOF 预测和三个完整基础 checkpoint；只用训练标签拟合融合器。模型在任意新目录下的官方验证可沿用上面的预测/评分命令，更换 checkpoint 和 output。`evaluate_stage4.py` 固定评估原 `stage4_final`，已有结果时拒绝覆盖。

`STAGE4_FILE_MANIFEST.json` 保存研究完成时的 SHA-256 快照（排除自身与 Python 缓存）。之后新增复现文件不属于原快照。原 V1/前三步清单及交付包保持不变。
''',encoding='utf-8')
    (ROOT/'STAGE4_AI_DISCLOSURE.md').write_text('''# 第四步 AI 协助披露

用户授权 AI 助手继续研究算术一致性。助手实现有界数值解析、输入软特征、匹配对照、自动检查与官方验证，并撰写分析。未使用外部模型/数据、候选生成或联网推理，依赖仍为 Python/NumPy。

计划在第四步验证前固定，三个模式/三个种子/L2=0.001 不因成绩扩展。训练输入检查修复百分比片段和百分数单位歧义；中间一次编码错误被检查发现并修复，日志保留。验证前百分号修正取代了最初九次拟合，实际共18次拟合，只有最终九组接受官方验证。没有挑选旧版本成绩。

基础 OOF 预测和完整基础模型复用第三步；新增标准化与融合权重仅拟合训练输入/标签。验证用于开发选型和诊断；其结果与不确定区间不构成新盲测，也不是经选择校正后的泛化保证。

训练解析例子和验证案例由 AI 助手辅助核查，不能称独立人工审核。最终标签只判断官方解析的最终答案，不能据此声称推理正确。算术不一致是软证据，未作为硬否决规则。

未运行新测试预测/评分或最终提交。独立完整性哈希核查读取旧文件字节（包括历史测试文件），但不解码或分析测试题目/标签；V1和前三步所有原产物保留。
''',encoding='utf-8')
    save(RUN/'implementation_history.json',{'pre_validation_revisions': [
        'Initial numeric-fragment review; preserve stage4_parser',
        'Prefix guard placement bug: 7 unit-test errors, fixed before training; failure log preserved',
        'First frozen audit plus 9 fits in stage4, never validated',
        'Train-input review found percent-answer unit ambiguity; final parser skips such RHS',
        'Final parser_v2 audit plus 9 final fits before any stage4 validation'],
        'total_fusion_fits':18,'final_fusion_fits':9,'official_validation_runs':9,
        'measured_training_entry_seconds':measured_training,'measured_complete_parser_audit_seconds':measured_parser,
        'unmeasured_failed_parser_audits':1,'new_base_fits':0,
        'final_parser_sha256':parser_train['parser_sha256'],'unit_tests_passed':27,
        'source': 'Recorded tool output and saved training/audit JSON; no inference weights changed after validation.'})
    save(RUN/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'cpu':'Intel Core i7-14700K','thread_limits':4,'parallel_workers':1,'new_dependencies':False})
    print(json.dumps({'report':str(ROOT/'STAGE4_RESULTS.md'),'preservation':checks,'primary':primary,
                      'measured_training_seconds':measured_training,'measured_parser_seconds':measured_parser},indent=2))


def write_cases():
    cases=read('changed_choice_examples.json')
    descriptions={
        23: ('修正总和错误','原候选写 3+5+7+9+11=36，检查器算出35；新候选写35。原错式与新正确式提供了直接可读的证据，但预测选择仍受全部融合特征影响。'),
        49: ('修正货币换算后的加法','原候选把2400+480+33写成2853，实际2913；新候选2913被选中。单位转换本身并非检查器验证的能力。'),
        111: ('反例：算对营业收入但遗漏利润','原候选最终308正确，却因单位/转义货币格式没有可检查式。新候选112×6=672和28×35=980都算对，却把营业收入980当作收益。算术正确不能识别漏减成本。'),
        114: ('修正小费求和','原候选56+168+72=300，实际296；新候选2×148=296。'),
        141: ('反例：正确最终答案含错误推理','官方正确最终答案为7，但原候选错误换算8盎司并写1+1+1+2+1.5+1=7（该左侧实际7.5）。新候选6.5无可检查式，模型改错。最终答案标签与整段推理有效性并不等价。'),
        219: ('两段恢复时间：语义遗漏','原候选4+2=6算对，却只回答第二阶段；新候选4+6=10回答总恢复时间。两者可检查式都正确，不能将改善单独归因于发现算术错误。'),
        255: ('利息与舍入/题意','原候选5600+392=6000算错，并把第二年本金变成6000；新候选按原本金计息得到6384。检查器识别加法错误，未直接验证简单利息语义。'),
        267: ('修正乘法错误','原候选15×58940=883100，实际884100；新候选使用884100，最终女性人数377050正确。'),
        287: ('香水：遗漏瓶数及覆盖局限','原候选只算一瓶需40丛，忽略20瓶；新候选800正确。原候选单位穿插导致没有可检查式，新候选仅20×12=240可检查。该例不能证明算术检查器能独立发现漏乘瓶数。')}
    lines=['# 第四步逐题案例核查','',
           '范围：最终主种子 control 与 arithmetic 的正确性发生变化的9题，包含全部7题修正与2题损失。按验证顺序记录，不只展示成功例。AI辅助分析，不是独立人工审核；未用于追加规则或重新训练。完整候选文本和提取等式保存在 `code/runs/stage4_final/changed_choice_examples.json`。','']
    for index,(title,body) in descriptions.items():
        e=cases[index]
        lines += [f'## {title}', '', f"题号：`{e['id']}`；候选 {e['control_index']} → {e['arithmetic_index']}；官方最终标签 {e['control_correct']} → {e['arithmetic_correct']}。",'',body,'']
    (ROOT/'STAGE4_CASE_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
