"""Apply fixed gates and write an honest local research report."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="4"
from pathlib import Path
import numpy as np
from experiment_io import ROOT,PROJECT,PUBLICATION,read,save,sha,utc,guard,snapshot_original
from provenance import snapshot

def main():
    opened,denied=guard("analysis");r=ROOT/"runs/experiment"
    extra=2 if (ROOT/"runs/attempt0_stopped/training-launch.log").exists() else 0
    t=read(r/"training_complete.json");v=read(r/"validation/summary.json");audit=read(ROOT/"verification/pre_validation_audit.json")
    assert audit["before_validation"] and audit["deployment_matches_training_scores"]
    rec=v["records"];modes=[m["mode"] for m in t["models"]];decision={}
    for m in t["models"]:
        mode=m["mode"];cv=[c for c in t["cv"] if c["mode"]==mode]
        checks={"two_cv_partitions_improve":all(c["disagreement"]["correct"]>t["baseline_cv"][str(c["seed"])]["disagreement"]["correct"] for c in cv),
          "validation_hard_at_least_309":rec[mode]["disagreement"]["correct"]>=309,
          "validation_overall_at_least_748":rec[mode]["all"]["correct"]>=748,
          "optimizer_or_fixed_rounds":audit["reference_optimizer"]["all_selected_successful"] if mode=="reference_full" else True,
          "asset_limit":m["uncompressed_bytes"]<=16*2**20,"input_contracts":mode in v["input_contracts"],
          "statistics_and_isolation_checks":audit["old_artifacts_unchanged"] and not t["denied"]}
        decision[mode]={"checks":checks,"eligible":all(checks.values()),"mean_cv_hard_correct":float(np.mean([c["disagreement"]["correct"] for c in cv]))}
    eligible=[m for m,z in decision.items() if z["eligible"]]
    selected=sorted(eligible,key=lambda m:(-decision[m]["mean_cv_hard_correct"],m!="reference_full"))[0] if eligible else None
    save(r/"DECISION.json",{"created_utc":utc(),"candidate_checks":decision,"eligible_candidate":selected,
       "test_evaluation_performed":False,"publication_changed":False,"thresholds_are_research_not_course_rules":True,
       "reference_CPU_eligibility_not_proven_by_local_timing":True})
    original=snapshot_original();assert original==read(r/"budget.json")["original"]
    prior=snapshot(True)
    pub=read(PUBLICATION/"PUBLICATION_MANIFEST.json");count=0
    for rel,meta in pub["files"].items():
        if any("test" in part.lower() or "audit" in part.lower() for part in Path(rel).parts):continue
        assert sha(PUBLICATION/rel)==meta["sha256"],rel
        count+=1
    for m in t["models"]:
        for name,digest in m["sha256"].items():assert sha(ROOT/m["checkpoint"]/name)==digest
    save(ROOT/"verification/final_audit.json",{"finished_utc":utc(),"original_frozen_and_report_unchanged":True,
        "previous_files_verified":sum(z["files_verified"] for z in prior.values()),"publication_non_test_files_verified":count,
        "deployment_hashes_unchanged":True,"no_test_evaluation":True,"no_model_or_report_publication":True,
        "actual_SciPy_calls_including_stopped_run":66+extra,"actual_tree_fits":61,"actual_training_PCA_fits":11,
        "synthetic_gradient_and_two_PCA_invariance_checks_not_training_search":True})
    def label(m):
        if m.startswith("tree_"):
            return ("浅层树：人工47维" if "_hand_" in m else "浅层树：人工47维＋PCA32维")+"，深度"+m[-1]
        return {"reference_full":"完整参考配置","ce_baseline":"当前CE参照","frozen_v1_1":"已发布冻结方案"}.get(m,m)
    lines=["# 错误诊断与有限优化对照结果","",
      "日期：2026-10-30。本轮完成训练折外错误诊断、完整参考配方和六种浅层树配置的有限比较。再次测试资格按训练前固定的门槛判断，详见DECISION.json。没有测试跑分、成绩提交或发布版本改动。","",
      ("结论：没有候选达到再次测试门槛，保留现有提交并结束本轮追加搜索。" if selected is None else f"结论：{selected}通过本轮研究门槛，取得后续冻结和单次测试的研究资格；本轮尚未执行测试。"),
      "","## 1. 错误诊断提供了什么信息","",
      "诊断仅使用train与既有完整折外预测。两条主支路任一答对的标签oracle在两个划分分别为1863/2114（88.13%）、1869/2114（88.41%）；六方法oracle为1905/2114、1904/2114。它们使用真实标签，不能部署，不能当作新模型成绩或可保证达到的目标。两条主支路在两个划分中均错误的题共205道。","",
      "主划分1086次候选索引差异中823次最终可解析答案相同，真正答案变化及不可解析部分仅263次。因此不能把索引差异当作有效互补。","",
      "按预案抽样28题并逐题AI辅助复核：16道稳定共同错误、8道互补题、4道答案等价对照。错误包括漏掉人数/倍率、读反增减关系、总量与剩余量混淆、单位/计数粒度及整数购买约束等。正确候选也可能有明显中间错误；样本还存在参考答案口径与自然语言理解歧义。保留官方标签，不将AI解释写成替代真值。详见[诊断报告](diagnosis/DIAGNOSIS.md)。","",
      "## 2. 固定对照与选择边界","",
      "完整参考配置使用81维人工特征、1024维原表示、1024维组中心化表示，仅训练分歧题并用其统计标准化、裁剪±20，三个分块L2成员取平均。特征与目标函数来自固定公开快照，保留已知grounding类型问题；未引入对方权重。来源与已知实现限制见SOURCE_NOTE.md。","",
      "六种树使用已有47维人工特征，或再加各折训练池独立拟合的32维PCA表示；深度2/3/4，LambdaRank，固定200轮，其余参数一致。没有新语义规则、验证早停或按验证结果选择树。训练OOF按平均分歧题答对数、最差划分、整体、浅度及少特征依次排序。","",
      "树的训练选中配置："+t["tree_selected"]+"。其余五个树不进入验证评分。所有成员在验证开始前固定配置和检查点哈希。","",
      "## 3. 两组训练折外交叉验证","",
      "| 配置 | 划分20261022：分歧正确/2114 | 划分20261029：分歧正确/2114 |","|---|---:|---:|",
      f"| CE参照 | {t['baseline_cv']['20261022']['disagreement']['correct']} | {t['baseline_cv']['20261029']['disagreement']['correct']} |"]
    for mode in ("reference_full","tree_hand_d2","tree_hand_d3","tree_hand_d4","tree_pca32_d2","tree_pca32_d3","tree_pca32_d4"):
        z={c["seed"]:c for c in t["cv"] if c["mode"]==mode}
        lines.append(f"| {label(mode)} | {z[20261022]['disagreement']['correct']} | {z[20261029]['disagreement']['correct']} |")
    lines+=["","## 4. 四次官方验证","",
      "| 配置 | 全部798题 | 分歧355题 | 其他443题 |","|---|---:|---:|---:|"]
    for mode,z in rec.items():
        lines.append(f"| {label(mode)} | {z['all']['correct']}/798（{z['all']['accuracy']:.2f}%） | {z['disagreement']['correct']}/355（{z['disagreement']['accuracy']:.2f}%） | {z['other']['correct']}/443 |")
    lines+=["","相对CE的配对变化：","",
      "| 配置 | 修复 | 损坏 | 分歧提升百分点 | 描述性95%配对区间 |","|---|---:|---:|---:|---|"]
    for mode,p in v["paired"].items():
        q=p["disagreement"];ci=q["descriptive_paired_bootstrap_95_interval"]
        lines.append(f"| {label(mode)} | {q['corrected']} | {q['broken']} | {q['gain_percentage_points']:+.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines+=["","这些区间仅作描述，未校正多轮比较；反复使用的验证集不是新的盲测。当前已提交方案历史测试82.23%不能替代新候选未知的测试成绩。","",
      "## 5. 优化停止状态与技术偏差",""]
    a=audit["reference_optimizer"]
    lines+=[f"完整运行记录66次SciPy数值优化，其中{a['successful_starts']}次报告成功；33个按训练目标选中的成员中{a['selected_successful']}个成功停止。最大选中梯度无穷范数为{a['max_selected_gradient_inf']:.6g}。采用参考原600次迭代上限和默认相对目标停止容差，不能将它们描述为全部达到此前CE的1e-6梯度标准。","",
      ("首个运行因额外成功标记检查被中止，已保留目录、日志与源码；验证前修改编排检查，使其保留原参考优化设置并完整记录未成功状态。没有提高迭代上限或添加配置。首次两个未保存权重的初值被重新运行，因此实际共68次SciPy调用，而非原计划66次。此差异发生在任何新验证评分前，不属于按验证结果追加搜索。详见TECHNICAL_NOTE.md。" if extra else "本次复现没有首轮中止的额外数值拟合；共66次SciPy调用。"),"",
      "参考配置的未成功停止限制了结论：这里只能判断原上限和实现下的表现，不能据此否定充分收敛版本的潜力。树按固定200轮完成，参数有限；完成固定轮数也不等同于优化收敛。","",
      "## 6. 门槛判断与后续范围","",
      "研究门槛保持不变：两组训练CV分歧题均严格优于CE；验证分歧至少309/355，整体至少748/798；实现、隔离、资源与复现通过。参考还需通过成员停止状态检查。它们是本项目决策规则，不是课程规定的正确题数，也不保证测试提升。",""]
    for mode,z in decision.items():
        descriptions={"two_cv_partitions_improve":"两组训练CV均改善","validation_hard_at_least_309":"验证分歧题至少309题",
          "validation_overall_at_least_748":"验证整体至少748题","optimizer_or_fixed_rounds":"优化停止状态",
          "asset_limit":"资产限制","input_contracts":"输入接口","statistics_and_isolation_checks":"统计与隔离核查"}
        failed=[descriptions[k] for k,val in z["checks"].items() if not val]
        lines.append(f"- {label(mode)}："+("通过研究门槛。" if z["eligible"] else "未通过："+", ".join(failed)+"。"))
    lines+=["","本轮不追加蒸馏、融合、更多树超参数或人工规则。若没有候选达标，建议保留现有提交并整理负面实验；今后若继续，应提出能增加题目条件与推理对应信息的新假设，而非将现有特征重复拼接。诊断只表明这些问题值得研究，没有证明任何新文本方法一定有效。","",
      "## 7. 成本、资源和复现","",
      f"完整运行：33个参考成员（66个优化初值）＋61个树模型＝94个训练成员/树，共127次优化/树拟合调用；另有11个训练PCA变换。中止运行额外{extra}个初值，合计{127+extra}次数值优化/树调用。完整训练墙钟{t['training_wall_seconds']:.3f}秒；其中参考与树拟合累计{t['optimizer_and_tree_seconds']:.3f}秒，PCA累计{t['pca_seconds']:.3f}秒。额外中止运行耗时未计入完整运行耗时，故该数值不是全部研究总时间。","",
      "独立环境：Python3.12.14、NumPy2.5.3、SciPy1.18.0、LightGBM4.7.0，其他依赖见environment-packages.txt。未改变原项目环境。机器为i7-14700K，模型训练和验证各四线程、单模型工作进程。以下是fresh预测＋独立官方评分的本机完整validation耗时，不替代参考Xeon CPU资格：","",
      "| 配置 | 未压缩推理资产字节 | 预测＋评分秒 |","|---|---:|---:|"]
    for mode,z in rec.items():lines.append(f"| {label(mode)} | {z['uncompressed_bytes']} | {z['wall_seconds']:.3f} |")
    lines+=["","11组参考标准化参数按实际训练行重算逐元素一致；11组PCA核对训练均值、轴正交性和训练协方差特征方程；10组训练/留出行集合隔离、60个CV树的轮数和模型哈希通过。部署分数与训练入口一致，并通过新ID、空输入、1/3/7题批次与候选置换检查。梯度有限差分和PCA留出输入变更不影响拟合的合成检查通过。","",
      "旧五轮研究共1006个工件、原冻结模型/报告、发布清单及504个非测试发布文件保持一致；新检查点哈希在验证后仍一致。官方数据、数值解析器与评分器不变。本轮没有读取测试输入/参考答案或新增测试评分，也没有GitHub发布。","",
      "代码、统计、折行号、全部OOF、部署文件、优化停止状态与官方验证输出均保留。复现步骤见REPRODUCE.md，工件哈希见ARTIFACT_MANIFEST.json。","",
      "AI assistance：OpenAI Codex实质协助本轮错误诊断、案例复核、代码实现、环境安装、实验编排与执行、统计分析及文档。案例复核不是独立人工审查，也不声称学生亲自运行本轮全部命令。无外部训练数据、额外预训练权重、API教师或新候选生成；保留课程数据来源与许可归属，公开参考代码复用已在SOURCE_NOTE.md披露。"]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Final decision:",selected,"; previous artifacts:",sum(z["files_verified"] for z in prior.values()))
if __name__=="__main__":main()
