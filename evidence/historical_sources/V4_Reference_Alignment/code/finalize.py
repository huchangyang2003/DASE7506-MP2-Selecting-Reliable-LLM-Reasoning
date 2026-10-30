"""Finish audits and describe the preregistered decision; never evaluates test."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
from pathlib import Path
import numpy as np
from experiment_io import ROOT,DATA,PUBLICATION,read,save,sha,guard,utc,snapshot_original
from provenance import snapshot
from common import load_inputs
from linear_core import build_inputs,fit_scaler
from scope import MODES,fit_and_scaler_rows
from experiment_io import disagreement

def main():
    opened,denied=guard("analysis")
    r=ROOT/"runs/scope";t=read(r/"training_complete.json");v=read(r/"validation/summary.json")
    bank,arrays=load_inputs("train",DATA);hard=disagreement(bank);x,_=build_inputs(bank,arrays)
    rows=read(r/"row_audit.json");assert len(rows)==30
    for row in rows:
        pool=np.asarray(row["pool"]);held=set(row["held"]);f,s=fit_and_scaler_rows(pool,hard,row["mode"])
        assert f.tolist()==row["fit"] and s.tolist()==row["statistics"]
        assert not (set(f)&held or set(s)&held)
    for m in t["models"]:
        _,stats=fit_and_scaler_rows(np.arange(len(bank)),hard,m["mode"])
        mean,scale=fit_scaler(x[stats])
        with np.load(ROOT/m["checkpoint"]/"model.npz",allow_pickle=False) as z:
            assert np.array_equal(z["mean"],mean) and np.array_equal(z["scale"],scale)
        for name,digest in m["sha256"].items():assert sha(ROOT/m["checkpoint"]/name)==digest
    original=snapshot_original();assert original==read(r/"budget.json")["original"]
    prior=snapshot(True)
    pub=read(PUBLICATION/"PUBLICATION_MANIFEST.json");verified=0
    for rel,meta in pub["files"].items():
        if any(("test" in p.lower() or "audit" in p.lower()) for p in Path(rel).parts):continue
        assert sha(PUBLICATION/rel)==meta["sha256"],rel
        verified+=1
    fits_converged=all(f["converged"] for f in t["fits"])
    assert fits_converged and len(t["fits"])==33
    cv={(c["mode"],c["seed"]):c for c in t["cv"]};rec=v["records"];decision={}
    for mode in MODES[1:]:
        checks={"both_cv_hard_better":all(cv[mode,s]["disagreement"]["correct"]>cv["baseline",s]["disagreement"]["correct"] for s in (20261022,20261029)),
                "validation_hard_at_least_309":rec[mode]["disagreement"]["correct"]>=309,
                "validation_all_at_least_748":rec[mode]["all"]["correct"]>=748,
                "all_optimizers_converged":fits_converged,"interfaces_and_official_evaluation":v["input_contracts_passed"],
                "asset_below_16MiB":rec[mode]["uncompressed_bytes"]<=16*1024**2}
        decision[mode]={"checks":checks,"eligible":all(checks.values()),"mean_cv_hard_correct":float(np.mean([cv[mode,s]["disagreement"]["correct"] for s in (20261022,20261029)]))}
    eligible=[m for m,d in decision.items() if d["eligible"]]
    chosen=sorted(eligible,key=lambda m:(-decision[m]["mean_cv_hard_correct"],m!="hard_all_scaler"))[0] if eligible else None
    save(r/"DECISION.json",{"created_at_utc":utc(),"criteria_set_in_PLAN_before_training":True,"candidates":decision,"eligible_candidate":chosen,
         "test_evaluation_performed":False,"publication_changed":False,"research_threshold_not_course_rule":True,
         "reference_cpu_compliance_not_established_by_local_timing":True})
    save(ROOT/"verification/final_audit.json",{"created_at_utc":utc(),"row_audits":30,"full_scalers_recomputed_exact":3,
         "fits_converged":33,"original_and_prior_unchanged":True,"prior_artifact_files":sum(p["files_verified"] for p in prior.values()),
         "publication_non_test_files_verified":verified,"checkpoint_hashes_unchanged":True,"source_data_opened":sorted(opened),
         "no_test_evaluation":True,"denied":denied})
    names={"baseline":"全训练题 CE 基线","hard_all_scaler":"仅分歧题训练＋全训练池标准化","hard_hard_scaler":"仅分歧题训练＋分歧题标准化"}
    lines=["# 参考方案对齐：阶段 A——训练范围与标准化范围","",
    "日期：2026-10-29。本轮没有达到再次测试跑分的预设门槛，保留现有冻结方案。分歧题训练仅带来验证集净增 1 题；同步改用分歧题标准化时，整体出现明显退步。这不等同于完整参考配置无效，因为本轮尚未复现其完整特征、裁剪与分块正则化组合。","",
    "## 1. 预先固定的对照","",
    "保留 1071 维原有特征（1024 维表示＋30 维基础与组内统计＋17 维原算术特征），统一使用 uniform-positive softmax CE、L2=0.01、float64 L-BFGS。三个方案仅改变哪些训练题参与损失，以及哪些训练题用于估计均值、标准差。分歧题按输入候选中至少两个不同的可解析答案确定，不依赖正确答案。","",
    "| 方案 | 每折标签拟合范围 | 每折标准化统计范围 |","|---|---|---|"]
    for m in MODES:lines.append("| "+names[m]+" | "+("所有训练池题" if m=="baseline" else "训练池中的分歧题")+" | "+("训练池中的分歧题" if m=="hard_hard_scaler" else "所有训练池题")+" |")
    lines+=["","全量训练共 4755 题，其中分歧题 2114 题。每组五折划分均以完整题目为单位，四个候选始终一起进入训练或留出集；标准化只使用该折训练池。仅分歧题方案按实际参与拟合的题数对损失取平均，故去除其他题也改变了损失曲率和相对于 L2 的尺度；不能将结果简单归因于标签更干净。","",
    "## 2. 两组训练交叉验证","",
    "下表是各自训练划分的 OOF 分歧题答对数，分母均为 2114。种子是预先固定的划分种子；线性优化没有用多个随机初始化择优。","",
    "| 方案 | 划分 20261022 | 划分 20261029 |","|---|---:|---:|"]
    for m in MODES:lines.append(f"| {names[m]} | {cv[m,20261022]['disagreement']['correct']}/2114 | {cv[m,20261029]['disagreement']['correct']}/2114 |")
    lines+=["","仅分歧题训练且保留全训练池标准化，在两组划分中分别相对基线 -1、+6 题，方向不一致。分歧题标准化分别 +1、+1 题，但后面的独立验证暴露了明显的其他题退化。","",
    "## 3. 官方验证集结果","",
    "所有 33 次拟合完成并记录检查点哈希后，才读取验证集。通过未改动的官方预测和评分程序评估四个方法，未读取或评估本轮测试集。","",
    "| 方案 | 全部 798 题 | 分歧 355 题 | 其他 443 题 |","|---|---:|---:|---:|"]
    for m in ("frozen_v1_1",)+MODES:
        a=rec[m];lines.append(f"| {names.get(m,'已发布冻结方案')} | {a['all']['correct']}/798（{a['all']['accuracy']:.2f}%） | {a['disagreement']['correct']}/355（{a['disagreement']['accuracy']:.2f}%） | {a['other']['correct']}/443 |")
    lines+=["","相对本轮 CE 基线，第一种改动修复 8 题、损坏 7 题，净增 1 题。分歧题提升为 0.282 个百分点，描述性配对 bootstrap 95% 区间为 [-1.972, 2.261] 个百分点，包含零。这是小幅且不稳定的变化，不能据此声称已提高真实测试准确率。区间没有对多轮探索进行校正。","",
    "## 4. 标准化退步的核查","",
    "分歧题标准化方案新增的 55 道其他题错误，全部选择了官方解析器无法解析的候选；另外在分歧题中相对 CE 基线净少答对 1 题。两项合计造成整体少答对 56 题。CE 基线及保留全训练池标准化的方案均未选择不可解析候选。","",
    "输入统计诊断发现，different_answer_cosine 在全训练集上的标准差为 0.48974，而分歧题训练子集上仅为 0.01434，相差约 34.16 倍。标准化后，这类特征在不同题型上可能产生较大分布变化，也会改变 L2 对原始特征系数的有效约束。这提供了原因线索，尚未通过单特征消融确立因果关系。没有在看到验证结果后追加有效性过滤或裁剪规则来挽救本轮配置。","",
    "## 5. 何时再次测试跑分","",
    "本轮训练开始前，PLAN.md 已固定以下研究决策门槛，均非课程硬性要求：","",
    "1. 同一候选在两组训练交叉验证的分歧题上都优于当前 CE 基线；","2. 验证集分歧题至少 309/355（87.04%），相对当前 CE 的 305/355 至少净增 4 题；整体至少 748/798；",
    "3. 优化收敛、训练隔离、推理接口、官方评分与复现检查通过，模型资源满足作业限制；本机耗时不能替代课程参考 CPU 的资格判定；",
    "4. 达标后只冻结一个完整候选，再安排一次最终测试。多个候选达标时按预先规定的平均训练 CV 分歧题答对数选择，并列优先较简单方案。不能根据测试分数回头挑选配置。","",
    "仅分歧题训练＋全训练池标准化不满足双划分一致改善及 309 题门槛；分歧题标准化不满足验证分歧题和整体门槛。没有候选取得再次测试资格。本轮不修改已提交的模型、报告或 GitHub 仓库。测试集历史分数已知，后续测试不能称为首次盲测；验证集也经过多轮探索，上述门槛不保证测试提升。","",
    "## 6. 下一阶段范围","",
    "下一阶段应先冻结一份完整参考配置复现计划，单独对照参考实现的 81 维人工特征、原始与组内中心化表示、±20 标准化裁剪和分块正则化组合。不能把本轮两种范围改动称为完整复现，也不能将标准化的本轮失败推广为整个参考方法无效。若完整配置确有改善，再做预先固定的消融以定位来源，而非立即添加更多融合成员。","",
    "## 7. 运行与完整性记录","",
    f"共 33 次拟合全部收敛，优化累计 {t['optimization_seconds']:.3f} 秒，训练阶段墙钟 {t['wall_seconds']:.3f} 秒。本机为 Intel Core i7-14700K，BLAS 4 线程、单任务工作进程。四次官方验证的预测加评分墙钟分别为："]
    for m in ("frozen_v1_1",)+MODES:lines.append(f"- {names.get(m,'已发布冻结方案')}：{rec[m]['wall_seconds']:.3f} 秒；检查点文件合计 {rec[m]['uncompressed_bytes']} 字节。")
    lines+=["","基线 OOF 分数、折划分、全量权重以及验证选择与第一轮 CE 逐项一致。30 组实际折行集审计与 3 个全量标准化参数重算通过；新题目 ID、7 题批次和候选置换接口检查通过。旧四轮共 899 个工件的哈希、原冻结文件、原报告、发布清单及已核查的非测试发布文件均保持一致。官方数据、解析器、预测器与评分器没有修改。","",
    "主要证据：runs/scope/budget.json、training_complete.json、row_audit.json、validation/summary.json、DECISION.json，以及 verification/ 下的检查记录。ARTIFACT_MANIFEST.json 保存本轮工件哈希。","",
    "AI assistance：OpenAI Codex 实质性协助了本轮实现、命令执行、实验编排、统计计算、结果解释与文档撰写。本轮自动执行的实验不表述为学生亲自独立运行。课程提供的候选表示和原有数据保持不变，未使用新模型权重、外部训练数据或新增候选生成。"]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Final audits passed; eligible candidate:",chosen)
if __name__=="__main__":main()
