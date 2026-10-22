"""Record stage-3 results, reproducibility notes and immutable-ancestor checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'code/runs/stage3'


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def preservation():
    checks = {}
    v1 = ROOT.parent / 'DASE7506-MP2-main'
    for name, base, manifest in [('v1', v1, 'PROJECT_FILE_MANIFEST.json'),
                                  ('v2_stage1', ROOT, 'FILE_MANIFEST.json'),
                                  ('v2_stage2', ROOT, 'STAGE2_FILE_MANIFEST.json')]:
        entries = json.loads((base / manifest).read_text(encoding='utf-8'))['files']
        for filename, record in entries.items():
            assert hashlib.sha256((base / filename).read_bytes()).hexdigest() == record['sha256'], filename
        checks[f'{name}_original_entries_unchanged'] = len(entries)
    archives = {}
    for line in (v1 / 'deliverables/SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split()
        archives[name] = hashlib.sha256((v1 / 'deliverables' / name).read_bytes()).hexdigest() == digest
        assert archives[name]
    checks.update(v1_archives_unchanged=archives,
                  v2_test_files_present=[str(p.relative_to(ROOT)) for p in (ROOT / 'code/data').rglob('*test*')],
                  purpose='Integrity hashing only; original test bytes not decoded or analyzed, never used for fitting/selection.')
    assert not checks['v2_test_files_present']
    save(ROOT / 'STAGE3_PRESERVATION_CHECK.json', checks)
    return checks


def main():
    training, rows, aggregates = read('training_complete.json'), read('summary.json'), read('aggregates.json')
    selection, comparisons, verification = read('selection.json'), read('paired_comparisons.json'), read('inference_verification.json')
    checks = preservation()
    primary = next(r for r in rows if r['mode'] == selection['best_average_mode'] and r['seed'] == 20261022)
    comparison = comparisons[selection['best_average_mode']]['disagreement']
    aggregate_table = '\n'.join(f"| {r['mode']} | {r['accuracy_mean']:.2f} ± {r['accuracy_sample_std']:.2f}% | "
                                f"{r['disagreement_mean']:.2f} ± {r['disagreement_sample_std']:.2f}% |" for r in aggregates)
    seed_table = '\n'.join(f"| {r['mode']} | {r['seed']} | {r['correct']}/798 ({r['accuracy']:.2f}%) | "
                           f"{r['disagreement_correct']}/355 ({r['disagreement_accuracy']:.2f}%) |" for r in rows)
    strata_table = '\n'.join(f"| {r['distinct_valid_answers']} | {r['n']} | {r['mlp_control']:.2f}% | {r['stacking_0.001']:.2f}% |"
                             for r in read('strata.json'))
    paired_table = '\n'.join(f"| {r['seed']} | {r['corrected']} | {r['broken']} | {r['corrected'] - r['broken']:+d} |"
                             for r in comparison['paired_counts_by_seed'])
    elapsed = [r['total_wall_seconds'] for r in rows]
    training_seconds = training['total_training_wall_seconds']
    base_seconds = sum(r['seconds'] for r in training['new_base_fits'])
    meta_seconds = sum(r['seconds'] for r in training['meta_fits'])
    report = f'''# 第三步结果：训练集 OOF 融合带来验证集改善

第三步已完成，预设 21 个新基础模型拟合、9 个融合器拟合、15 组官方验证全部完成。选中 **OOF stacking，L2=0.001**。三个种子的完整验证平均准确率由 91.60% 提高到 **92.56%**，分歧题由 81.13% 提高到 **83.29%**；三个种子均有改善。没有进行新的测试集预测、评分或最终提交，V1 原模型保持不变。

## 设计与防止数据泄漏

使用既有的 MLP32 BCE、embedding 逻辑回归（固定 L2=0.01）与答案支持票数。既有 MLP 全训练 checkpoint 按种子复用；embedding 模型在完整训练集重新拟合。未添加依赖、外部训练数据、模型权重或新 LLM 生成。

4,755 个训练题的空白规范化题目均唯一。用固定随机种子生成五折，同题的四个候选保持在一起；若存在重复题目也会分到同一折。每折分别拟合 embedding 标准化及逻辑回归，以及 MLP 输入标准化和三个种子的 MLP。每道题的 OOF 预测仅来自没有见过该题的基础模型。五折的训练/留出题索引和无题目交集检查已保存。

融合器的训练输入是 **训练集 OOF 预测**，监督标签也是训练标签。训练进程实际只打开训练题库与训练参考答案；额外审计钩子拒绝验证文件和测试文件。全部基础和融合拟合于 {training['finished_at_utc']} 完成后，独立评估入口才加载验证集。验证集用于选模式，未参与融合权重、标准化或基础权重的拟合。

OOF stacking 是一个共享候选逻辑回归，保留每题多个正确候选标签。8 个特征是两个题内中心化 logit、中心化答案支持比例、两个同答案平均 logit、两个 logit×支持比例交互、中心化解析有效性。logit 固定截断到 [-30,30]，再构造题内特征；融合特征标准化仅拟合训练 OOF。所有特征只依赖传入题目、候选、embedding 和基础模型分数，没有题号或候选位置查表。

这轮同时引入了分数融合与同答案分数汇总/交互，不能把全部增益单独归因于“增加线性模型”。融合器已在 OOF 数据上拟合，因此 OOF 融合训练准确率不构成整个系统的无偏交叉验证估计。本轮不报告这样的误导性指标。

## 全部配置

下表为三个独立随机种子的平均值 ± 样本标准差，**不是跨种子推理集成**。固定平均是 sigmoid(MLP logit)、sigmoid(embedding logit)、答案支持比例三者等权平均。Stacking 的三个正则化强度在验证前已固定，没有追加搜索。

| 方法 | 完整验证集（798 题） | 分歧题（355 题） |
|---|---:|---:|
{aggregate_table}

| 方法 | 种子 | 完整正确题数 | 分歧正确题数 |
|---|---:|---:|---:|
{seed_table}

所有配置在另外 443 道非分歧题上都是 443/443。三个 MLP 对照的全部 798 个选择均与第一步同种子 BCE 预测完全相同，可以将差异与这些匹配对照比较。

## 改善的大小与不确定性

选中的 stacking 在分歧题上平均提高 **{comparison['mean_gain_percentage_points']:.2f} 个百分点**，三个种子分别提高 2.54、1.41、2.54 个百分点，满足事前声明的“平均至少 +1 个百分点且至少两个种子改善”规则。

按题对三个种子的正确性差异取平均，再进行 5,000 次题目重采样，分歧增益的描述性 95% 区间是 **[{comparison['question_bootstrap_95_interval'][0]:.2f}, {comparison['question_bootstrap_95_interval'][1]:.2f}] 个百分点**。完整集平均增益 +0.96 个百分点，区间约 [0.13,1.80]。这些区间没有校正反复使用验证集及模式选择，不能视为新的独立测试证据，也不能保证测试准确率超过历史 78.8%。

| 种子 | 修正原来错题 | 把原来对题改错 | 净增加正确题 |
|---|---:|---:|---:|
{paired_table}

保留事前固定的主种子 20261022：**740/798 = 92.73%**，分歧 **297/355 = 83.66%**。没有选用单次成绩最高的种子。主种子比本轮匹配 MLP 净多对 9 题；比 V1 历史主模型的验证成绩 733/798 多 7 题。后者训练批次不同，仅作历史参考。

## 分层与剩余空间

| 不同有效答案数 | 验证题数 | MLP 对照三种子平均 | 选中融合三种子平均 |
|---|---:|---:|---:|
{strata_table}

增益主要来自 2、3 个答案的题目；**4 个答案的 20 题从 66.67% 降至 58.33%**。该小子集不能稳定估计泛化效果，但说明融合并未改善所有题型。没有据此追加分组规则或继续调参。

三个基础选择器的“至少一个选对”诊断上限为完整集 95.86%–95.99%、分歧题 90.70%–90.99%。计算这个上限需要验证金标，它不是可实现的模型成绩，也未用于推理。融合器仍有未利用的互补空间。直接平均三种信号反而下降，说明不能仅因模型有互补性就假定任意组合都会改善。

## 成本、检查与复现

- 全部本轮训练入口耗时 **{training_seconds:.2f} 秒**，其中新基础拟合/折外打分/折模型落盘约 {base_seconds:.2f} 秒，9 个融合器优化约 {meta_seconds:.2f} 秒；不含复用的历史 MLP 训练成本，不含编写、测试和验证耗时。
- 主种子选中 checkpoint 未压缩资产 **{primary['inference_uncompressed_bytes']:,} 字节**（约 {primary['inference_uncompressed_bytes']/1024:.2f} KiB），小于 16 MiB；其中含两个完整基础模型、融合标准化、融合权重与配置，不依赖折模型或 OOF 文件进行推理。
- 主种子独立官方验证预测+评分进程耗时 **{primary['total_wall_seconds']:.3f} 秒**；15 次范围 {min(elapsed):.3f}–{max(elapsed):.3f} 秒。CPU/4 线程，本机验证集测量不等于参考 CPU 上测试集的正式资格认证。
- 17 项测试通过，包括五折同题隔离、训练折标准化、解析等价数、融合特征置换和逻辑回归梯度/收敛，以及前两步检查。
- 选中 checkpoint 重新加载的全部 798 个选择与官方输出一致；17 道题使用新题号仍一致；推理审计期间只读取 checkpoint 的 model.npz 与 config.json，并禁止网络访问。
- 候选重排后分数按相应候选映射一致；有 {verification['permutation_index_changes']} 道题因同分候选换了原索引，**解析答案变化 {verification['permutation_parsed_answer_changes']} 道**。不声称原候选索引严格不变。
- 首次评估在全部评分完成后，额外推理检查因 NumPy 断言工具的延迟导入被文件审计拦截而中止。已将断言工具在审计前导入，仅重跑推理检查，未重训、改权重、重评分或扩充配置。原失败日志及修复记录均保留。
- 完整性复核：V1 原 {checks['v1_original_entries_unchanged']} 项、第一步 {checks['v2_stage1_original_entries_unchanged']} 项、第二步 {checks['v2_stage2_original_entries_unchanged']} 项及两个原交付压缩包全部未改变。独立哈希检查包含旧测试相关文件的字节读取，但不解码、不分析测试题或标签。

选中开发 checkpoint：`code/{selection['development_checkpoint']}`。原 `submission.py` 和 V1 最终 checkpoint 均未替换，暂无新测试分数。

复现见 `STAGE3_REPRODUCE.md`；15 次官方评分、折划分、折模型、OOF 预测、训练记录、增益与逐题诊断均在 `code/runs/stage3/`；AI 协助披露见 `STAGE3_AI_DISCLOSURE.md`。

## 后续计划

第三步得到可保留的正向开发候选。下一步按原计划研究显式算术一致性特征，使用软证据而非硬否决，再与本轮融合候选做匹配比较；本轮尚未执行。若后续准备冻结最终方案，应先汇总所有开发阶段，再统一决定最终评估，避免反复用测试分数推动优化。
'''
    (ROOT / 'STAGE3_RESULTS.md').write_text(report, encoding='utf-8')
    (ROOT / 'STAGE3_REPRODUCE.md').write_text('''# 第三步复现

在 `V2_Ranking_Objective/code/` 中使用 V1 已配置的虚拟环境。只需要 Python 3.12.14 / NumPy 2.5.3；不联网、不安装新依赖。

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$py = '..\\..\\DASE7506-MP2-main\\code\\.venv\\Scripts\\python.exe'
```

仅重现选中模型的官方验证，不重训、不覆盖原评分：

```powershell
& $py -B run_official.py predict.py --method methods/fusion_model.py --checkpoint runs/stage3/stacking_0.001_seed20261022/checkpoint --split validation --output runs/stage3/reproduced-validation-predictions.json
& $py -B run_official.py score_predictions.py --split validation --predictions runs/stage3/reproduced-validation-predictions.json --output runs/stage3/reproduced-validation-score.json
```

期望 740/798（92.73182957393483%）。完整结果表中的 92.56% 是三个种子的平均，不是此单模型得分。

训练重现必须使用新目录，不覆盖历史产物：

```powershell
& $py -B train_stage3.py --output-dir runs/stage3_reproduction
```

训练会生成 21 个新基础模型拟合、9 个融合器拟合、15 个自包含部署 checkpoint；训练进程拒绝读取验证/测试文件。复用的全训练 MLP 来自现有 `runs/stage1/bce_seed*/checkpoint`。要评估新目录，可逐模型调用上面的官方验证命令并更换 checkpoint/output 路径；`evaluate_stage3.py` 固定评估原 stage3，已有结果时拒绝覆盖。

`evaluate_stage3.py --verify-only` 可单独复查原选中模型的保存加载、17 题新题号、候选重排和只读取 checkpoint 的推理约束。会更新同一验证检查记录；不拟合或重新评分。

`STAGE3_FILE_MANIFEST.json` 是本轮完成时的 SHA-256 快照。用户之后新增复现产物不会属于旧快照；已有产物不应覆盖。V1 的正式测试和交付包保持原样。
''', encoding='utf-8')
    (ROOT / 'STAGE3_AI_DISCLOSURE.md').write_text('''# 第三步 AI 协助披露

本轮由用户授权 AI 助手设计和实现训练集五折 OOF 融合，编写检查、运行 CPU 训练及官方验证，分析并记录全部预设结果。计划、融合特征、参数预算和判定规则在本轮验证结果产生前保存；没有根据结果追加搜索。

标准库与 NumPy 实现训练；`stage3_logistic.py` 为 V1 `train_linear.py` 的逐字节副本，仅复用其优化函数；第一步 MLP 拟合函数及已有全训练 BCE checkpoint 按原配置复用。未使用外部数据、预训练权重、付费接口或新增候选生成。

训练/融合权重拟合仅使用原始训练集；验证用于选择模式和诊断。OOF 标签被用于训练融合器，不能将同份 OOF 上融合器的拟合成绩称为无偏交叉验证结果。所有验证区间是重复使用开发集下的描述性分析。

未开展新测试预测、测试评分或最终提交。独立完整性检查仅读取历史文件字节进行哈希，包括旧测试相关文件，不解码分析其题目或标签。原 V1、前两步文件及归档全部保留。

推理检查首次在 NumPy 断言工具延迟导入时被严格文件审计中止；仅调整评估工具的预导入并重跑推理检查，未调整预测模型、训练设置或结果。日志保留。用户需要阅读分析并理解方法，不能将 AI 完成的检查描述为独立人工审核。
''', encoding='utf-8')
    save(RUN / 'verification_repair_record.json', {
        'original_evaluator_sha256': '7519a6283090e914786687b0da623f0e137dec967beb3cc0b6fbd203726bac98',
        'repaired_evaluator_sha256': hashlib.sha256((ROOT / 'code/evaluate_stage3.py').read_bytes()).hexdigest(),
        'failure_log': 'evaluation-launch.log', 'retry_log': 'verification-retry.log',
        'reason': 'Lazy import of NumPy testing helper occurred inside strict inference read guard',
        'repair': 'Preload assertion helper; add verify-only entry point',
        'refitted_models': 0, 'rescored_models': 0, 'predictions_changed': False, 'verification_passed': True})
    save(RUN / 'unit_test_summary.json', {'command': 'python -B -m unittest discover -s tests -v',
        'ran': 17, 'passed': 17, 'failed': 0, 'seconds': 0.079,
        'observed_before_stage3_training': True,
        'evidence': 'Tool-returned unittest output; five fusion checks plus twelve existing checks'})
    save(RUN / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'cpu': 'Intel Core i7-14700K', 'thread_limits': 4, 'parallel_training_workers': 1,
        'no_new_packages': True, 'official_scorer_unchanged': True})
    save(RUN / 'completion.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'new_base_fits': len(training['new_base_fits']), 'meta_fits': len(training['meta_fits']),
        'official_validation_runs': len(rows), 'inference_verification_passed': True,
        'test_evaluated': False, 'selected_primary': primary})
    print(json.dumps({'report': str(ROOT / 'STAGE3_RESULTS.md'), 'selection': selection,
                      'training_seconds': training_seconds, 'base_fit_seconds': base_seconds,
                      'meta_fit_seconds': meta_seconds, 'preservation': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
