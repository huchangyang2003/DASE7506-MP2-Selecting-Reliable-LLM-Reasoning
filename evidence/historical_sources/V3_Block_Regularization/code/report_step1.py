"""Read saved evidence and write the supplementary result/decision; no model fitting."""
import csv
from pathlib import Path
import numpy as np
from experiment_io import ROOT, FROZEN, PUBLICATION, read, save, sha, utc, guard, snapshot_original


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def main():
    opened, denied=guard("analysis")
    run=ROOT/"runs/step1"
    train=read(run/"training_complete.json")
    primary=read(run/"primary/summary.json")
    confirmation=read(run/"confirmation/summary.json")
    selected=read(run/"selection_before_confirmation_and_validation.json")["selected"]
    evaluation=read(run/"validation/summary.json")
    p={r["config"]:r for r in primary}
    c={r["config"]:r for r in confirmation}
    records=evaluation["records"]
    ce_name=next(s["config"] for s in selected if s["objective"]=="uniform_ce" and s["family"]=="uniform")
    old=records["frozen_v1_1"]; ce=records[ce_name]
    paired=evaluation["vs_frozen"][ce_name]["disagreement"]
    bce_uniform=next(s["config"] for s in selected if s["objective"]=="bce" and s["family"]=="uniform")
    bce_block=next(s["config"] for s in selected if s["objective"]=="bce" and s["family"]=="block")
    configured_checks=[]
    for s in selected:
        u=next(v["config"] for v in selected if v["objective"]==s["objective"] and v["family"]=="uniform")
        name=s["config"]
        checks={"primary_better_than_uniform":p[name]["disagreement"]["correct"]>p[u]["disagreement"]["correct"],
                "confirmation_better_than_uniform":c[name]["disagreement"]["correct"]>c[u]["disagreement"]["correct"],
                "validation_gain_at_least_1pp":records[name]["disagreement"]["accuracy"]-old["disagreement"]["accuracy"]>=1,
                "validation_overall_not_worse":records[name]["all"]["correct"]>=old["all"]["correct"]}
        configured_checks.append({**s,"criteria":checks,"passes_all":all(checks.values())})
    decision={"written_at_utc":utc(),"stage":"V3 step1","standalone_replacement_approved":False,
              "candidate_for_next_development":ce_name,"configuration":p[ce_name],
              "reason":"Uniform-positive CE is a promising alternative branch; block penalties add no established gain. Validation +3 questions is below the predeclared 1pp practical threshold and interval crosses zero.",
              "criteria":configured_checks,"frozen_v1_1_changed":False,"new_test_score":None,
              "new_github_publication":False,"next_step_not_started":True}
    assert not any(r["passes_all"] for r in configured_checks)
    save(run/"DECISION.json",decision)
    rows=[]
    for s in selected:
        name=s["config"]
        rows.append([s["objective"],s["family"],p[name]["embedding_l2"],p[name]["hand_l2"],
          f'{p[name]["disagreement"]["accuracy"]:.2f}%',
          f'{c[name]["disagreement"]["accuracy"]:.2f}%',
          f'{records[name]["all"]["correct"]}/798 ({records[name]["all"]["accuracy"]:.2f}%)',
          f'{records[name]["disagreement"]["correct"]}/355 ({records[name]["disagreement"]["accuracy"]:.2f}%)'])
    result_table=table(["目标","正则搜索","embedding L2","人工L2","主CV分歧","复查CV分歧","validation全体","validation分歧"],rows)
    comparison_rows=[]
    for e,h in ((.001,.001),(.01,.01),(.1,.1),(.01,.001),(.1,.001),(.1,.01),(.3,.001),(.3,.01)):
        b=next(r for r in primary if r["objective"]=="bce" and r["embedding_l2"]==e and r["hand_l2"]==h)
        t=next(r for r in primary if r["objective"]=="uniform_ce" and r["embedding_l2"]==e and r["hand_l2"]==h)
        comparison_rows.append([e,h,f'{b["disagreement"]["accuracy"]:.2f}%',f'{t["disagreement"]["accuracy"]:.2f}%',
                                f'{t["disagreement"]["accuracy"]-b["disagreement"]["accuracy"]:+.2f}'])
    matched_table=table(["embedding L2","人工L2","BCE主CV分歧","CE主CV分歧","CE-BCE（百分点）"],comparison_rows)
    reg_rows=[]
    for objective in ("bce","uniform_ce"):
        for e,h in ((.01,.001),(.1,.001),(.1,.01)):
            base=next(r for r in primary if r["objective"]==objective and r["embedding_l2"]==e and r["hand_l2"]==e)
            block=next(r for r in primary if r["objective"]==objective and r["embedding_l2"]==e and r["hand_l2"]==h)
            reg_rows.append([objective,e,h,f'{base["disagreement"]["accuracy"]:.2f}%',f'{block["disagreement"]["accuracy"]:.2f}%',
                            f'{block["disagreement"]["accuracy"]-base["disagreement"]["accuracy"]:+.2f}'])
    reg_table=table(["目标","固定embedding L2","改后人工L2","统一L2主CV分歧","分块主CV分歧","差值（百分点）"],reg_rows)
    cost_rows=[[name,f'{r["uncompressed_bytes"]:,}',f'{r["complete_validation_seconds"]:.3f}']
               for name,r in records.items()]
    cost_table=table(["方法","未压缩推理资产（字节）","本次完整validation耗时（秒）"],cost_rows)
    primary_counts={objective:sum(r["disagreement"]["accuracy"]>next(b["disagreement"]["accuracy"] for b in primary
                           if b["objective"]=="bce" and b["embedding_l2"]==r["embedding_l2"] and b["hand_l2"]==r["hand_l2"])
                     for r in primary if r["objective"]==objective) for objective in ("uniform_ce",)}
    text=f"""# 追加优化第一步：线性比较器与分块正则化

日期：2026-10-29。所有新实验仅使用训练与验证，不产生新测试集分数。

## 结论
得到一条值得保留的新分支：**均匀正类组内CE + 统一L2=0.01的线性选择器**。
它在validation达到748/798（93.73%），其中分歧题305/355（85.92%）；
原冻结融合系统为745/798（93.36%）和302/355（85.07%）。
新模型净增3题，分歧准确率增加0.85个百分点，**尚未达到预设至少1个百分点的替换门槛**。
分块正则本身没有得到独立收益：CE选出的最佳block配置恰好也是统一L2=0.01；
BCE分块候选反而弱于其统一正则对照。因此不能把本轮改善归因于“分块正则成功”。

结论是保留该CE模型作为下一阶段比较基线/潜在互补分支，继续保留原frozen-v1.1提交。
当前研究尚未进行新特征、融合、蒸馏、测试评分或GitHub上传。

## 1. 实验范围和固定条件
沿用已有1024维embedding、30维基本/风格/组特征及17维算术特征，共1071维。
算术按原融合方法题内中心化；不增加相对标准化副本、排名、交互或外部特征。
与原融合系统相比，新的直接线性结构发生变化，因此整体比较不能全部归因于损失。
用于隔离损失的证据是下面相同特征、相同L2下的CV对照。

4755个train题（2114个disagreement）全部等权参与；14353个正确候选标签全部保留。
原解析器、特征实现及评分器逐字节不变；没有外部数据、权重或候选生成。
5折始终按问题分组，空白规范化重复问题不跨折，按输入定义的分歧状态分层。
每折标准化只拟合该折训练候选，不裁剪。所有98次正式拟合达到1e-6梯度阈值。
这是凸目标的两个不同折划分复查，不是随机初始化稳定性实验。

BCE：所有候选平均二元交叉熵；CE：每题正确候选均匀目标的组内softmax交叉熵。
两者均加0.5乘分块加权平方范数，BCE截距不罚，CE共同截距固定0。
第一轮旧listwise采用正确候选总概率目标，与本次uniform-positive CE不同。
不同损失的归一化/梯度尺度不同，L2数值并不意味着相同统计收缩程度；
相同L2对照和各自有限网格选择应一起解释。

## 2. 预设搜索及选择
每种损失：统一L2取0.001、0.01、0.1；block的embedding取0.01、0.1、0.3，
人工块取0.001、0.01。18个候选标签中有2个重合，实际16个不同配置。
主折seed=20261022；按分歧正确数、全体正确数、较强embedding/人工L2的顺序确定各类候选。
选择写入selection_before_confirmation_and_validation.json后，用seed=20261029复查。
CE的两类赢家相同，故只有3个不同模型：80次主CV、15次固定候选复查、3次full-train拟合。

{result_table}

原冻结融合系统的validation参考为745/798和302/355，不能与上表训练OOF数字跨数据集直接排名。
本轮未为原融合系统重训嵌套OOF基线，也未把旧融合层的训练内分数当作无偏OOF。
第二次划分仍基于相同数据，且最终validation已反复开发使用；均不宣称无偏外推估计。

## 3. 损失函数的匹配对照
{matched_table}

相同特征和L2下，CE在{primary_counts["uniform_ce"]}/8个主CV配置优于BCE。
最优统一正则CE与BCE在主划分分歧准确率分别为{p[ce_name]["disagreement"]["accuracy"]:.2f}%和{p[bce_uniform]["disagreement"]["accuracy"]:.2f}%，
复查划分为{c[ce_name]["disagreement"]["accuracy"]:.2f}%和{c[bce_uniform]["disagreement"]["accuracy"]:.2f}%。
这支持“本设置下CE优于线性BCE”的有限结论，不能推广为任何网络和特征下都更优。
validation只评估按CV预选的模型，没有为每个固定L2额外运行验证。

## 4. 分块正则是否有用
{reg_table}

BCE最优统一正则为两块0.001；block网格没有embedding=0.001这一行。
因此两个搜索族的最优值差异也受预定搜索范围影响，不能据此否定所有分块方案。
在固定embedding惩罚的匹配行中可见部分差异，但没有转化为超过统一对照的所选赢家。
本轮不根据边界结果临时追加更小惩罚或扩大网格。

## 5. 新CE与冻结模型的逐题比较
在355个validation分歧题中：
- 两者都正确：{paired["both_correct"]}题；
- 新CE修正旧模型：{paired["corrected"]}题；
- 新CE破坏旧模型正确选择：{paired["broken"]}题；
- 两者都错误：{paired["both_wrong"]}题。
净收益为+3题。5000次按问题配对bootstrap的描述性95%区间为
[{paired["descriptive_paired_bootstrap_95_interval"][0]:.2f}, {paired["descriptive_paired_bootstrap_95_interval"][1]:.2f}]个百分点，跨越0；
这不是经多重搜索校正后的显著性检验。443个其他validation题两者全部正确。

两个模型正确集合的验证标签oracle并集为320/355，说明存在错误互补；
**它不是可以部署的选择器，也不是预计融合后能达到的准确率**。
不得据validation标签拟合门控或选择规则。是否能利用互补性需要另做训练OOF融合实验。
本阶段没有进一步逐题写规则或利用失败个例追加训练。

## 6. 资源
{cost_table}

以上均为同一台i7-14700K、4个BLAS线程、单模型worker，
每个方法一个新预测进程加独立评分进程，包含启动、导入、加载、预处理。
这是798题validation耗时，不是467题test或参考Xeon上的资格计时。
没有按题数线性外推测试耗时；没有用本次计时改写原报告历史测试计时。
新CE推理资产50,441字节，较原203,273字节减少约75.2%。
体积减少明确，但本次完整validation耗时基本相同，不能声称显著加速。

98次正式拟合优化累计{train["optimization_seconds"]:.3f}秒；
训练入口总墙钟{train["training_wall_seconds"]:.3f}秒（含特征、CV、重拟合、记录）。
这不包括实现、合成数值检查、报告和validation进程。
另有2次小型合成优化器检查，不属于真实数据配置搜索。无失败训练或不收敛正式拟合。

## 7. 决策及下一步
不替换原提交：validation净增3题、未达到预设门槛，区间跨0，
且分块搜索未优于对应统一正则对照。
保留uniform_ce_e0.01_h0.01作为下一阶段研究基线：
优先固定该损失与训练设置，添加少量真正的相对特征后作单因素对照；
本轮没有实施该下一阶段，也没有蒸馏。
未来若研究融合，应重新生成满足完整支路折外要求的TRAIN分数，
而不是在validation上拟合权重。主CV已用于选超参数，后续若复用也须如实披露选择偏差。

方法、权重及配置都在观察validation前完成冻结。原已知测试分数仅为历史结果，
本轮所有数据访问记录均无测试输入/标签/预测；新模型test准确率未知。
已提交冻结版本、正式报告、原仓库工作树与官方数据未修改。

## 8. 复现与证据
阅读PLAN.md、REPRODUCE.md；完整机器可读决策为runs/step1/DECISION.json。
主CV每折目标/梯度/停止原因、输入和源码SHA、fold分配、OOF分数、
全train checkpoint、官方validation预测/评分及访问日志均已保存。
RESULTS.md是本轮新增报告，不覆盖此前正式10页报告。

AI实质参与本轮设计、实现、数值检查、实验执行、结果分析与文档。
没有把模型推理称为独立人工案例审核，也没有声明用户亲自运行了本轮命令。
"""
    (ROOT/"RESULTS.md").write_text(text,encoding="utf-8")
    with (run/"primary_cv_grid.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["config","objective","embedding_l2","hand_l2","all_accuracy","disagreement_accuracy"])
        writer.writeheader()
        for r in primary:
            writer.writerow({"config":r["config"],"objective":r["objective"],"embedding_l2":r["embedding_l2"],
                 "hand_l2":r["hand_l2"],"all_accuracy":r["all"]["accuracy"],"disagreement_accuracy":r["disagreement"]["accuracy"]})
    # Protect originals without opening historical test/audit inputs or result records.
    expected=read(run/"budget.json")["original_artifacts_before"]
    actual=snapshot_original()
    assert actual==expected
    manifest=read(PUBLICATION/"PUBLICATION_MANIFEST.json")
    verified,skipped=[],[]
    for rel,metadata in manifest["files"].items():
        name=Path(rel).name.lower()
        if any("test" in part.lower() or "audit" in part.lower() for part in Path(rel).parts):
            skipped.append(rel)
            continue
        if sha(PUBLICATION/rel)!=metadata["sha256"]:
            raise AssertionError("Original publication changed: "+rel)
        verified.append(rel)
    copied=("common.py","features.py","arithmetic_features.py","fusion_core.py","predict.py","score_predictions.py","requirements.txt")
    assert all(sha(ROOT/"code"/name)==sha(FROZEN/"code"/name) for name in copied)
    provenance={"recorded_at_utc":utc(),"copied_frozen_files_sha256":{n:sha(FROZEN/"code"/n) for n in copied},
       "optimizer_basis":{"source":"V2_Ranking_Objective/code/stage3_logistic.py","sha256":sha(ROOT.parent/"V2_Ranking_Objective/code/stage3_logistic.py")},
       "original_freeze_manifest_unchanged":actual==expected,"original_frozen_files_verified":len(actual["frozen_files"]),
       "original_publication_non_test_files_verified":len(verified),"historical_test_paths_not_opened":skipped,
       "no_test_data_opened_by_reporting":True,"no_new_publication":True}
    save(ROOT/"verification/INTEGRITY.json",provenance)
    print({"report":str(ROOT/"RESULTS.md"),"frozen_files":len(actual["frozen_files"]),
           "publication_non_test_files":len(verified),"no_test_evaluation":True})

if __name__=="__main__":
    main()
