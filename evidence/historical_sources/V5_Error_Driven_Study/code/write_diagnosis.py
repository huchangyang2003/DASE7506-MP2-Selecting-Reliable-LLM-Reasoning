"""Write the train-only diagnostic report from fixed samples."""
from collections import Counter
from experiment_io import ROOT,read,sha,save
def main():
    folder=ROOT/"diagnosis";r=read(folder/"summary.json");review=read(folder/"CASE_REVIEW.json");cases=read(folder/"cases.json")
    assert [x["case"] for x in cases]==[x["case"] for x in review["entries"]]
    assert len(cases)==28 and r["stable_primary_both_wrong"]==205
    counts=Counter(tag for e in review["entries"][:16] for tag in e["tags"])
    lines=["# 训练折外共同错误诊断","",
      "日期：2026-10-30。本阶段没有新训练、验证或测试评分。结论仅来自4755道训练题及既有完整折外预测，其中2114道为分歧题。28道案例由AI辅助逐题复核，不称独立人工审核。","",
      "## 1. 总体证据","",
      "| 划分 | CE答对/2114 | 原系统与CE任一选对 | 六方法任一选对 |","|---|---:|---:|---:|"]
    for seed,z in r["results"].items():
        lines.append(f"| {seed} | {z['methods']['ce']['disagreement']['correct']} | {z['pair_oracle']['disagreement']['correct']}（{z['pair_oracle']['disagreement']['accuracy']:.2f}%） | {z['family_oracle']['disagreement']['correct']}（{z['family_oracle']['disagreement']['accuracy']:.2f}%） |")
    lines+=["","任一选对的oracle使用真实标签，只是事后诊断上界，不能部署、不能当预测成绩，也不保证存在能达到此数值的选择规则。六方法包括原完整系统、CE、相对排名、相对标准化、严格算术及仅分歧题训练＋全训练池标准化；它们共享数据与大量特征，不是六个独立教师。","",
      "两条主支路在两个划分中均错误的分歧题有205道。主划分两者共同错误251道，第二划分245道；因此“互补存在”与“仍有大量共同盲点”同时成立。","",
      "## 2. 索引变化与答案变化","",
      "| 划分 | 不同候选索引 | 其中可解析答案相同 | 其余答案变化或不可解析 | 原系统独对 | CE独对 |","|---|---:|---:|---:|---:|---:|"]
    for seed,z in r["results"].items():
        q=z["pairs"]["original__ce"]
        lines.append(f"| {seed} | {q['different_index_hard']} | {q['same_answer_different_index_hard']} | {q['different_answer_hard']} | {q['left_only_correct_hard']} | {q['right_only_correct_hard']} |")
    lines+=["","主划分1086次索引变化中823次仍是相同可解析答案。选择索引改变的数量不能直接代表模型互补性，更不能据此推断集成空间。原系统独对120题、CE独对121题，说明两者平均水平接近，部署难点是能否从输入可靠识别谁更可信。","",
      "## 3. 28道预先抽样案例","",
      "使用预先固定的20261030随机种子，无放回抽取16道稳定共同错题、8道主支路互补题、4道答案相同但索引不同的对照。下面类型允许多标签；样本经过条件筛选，不能将频率推广到所有训练错误。完整四候选及参考答案保存在cases.json，复核标签只用于诊断，不进入模型。","",
      "| 案例 | 分组 | 诊断标签 | 具体证据 |","|---|---|---|---|"]
    for case,e in zip(cases,review["entries"]):
        lines.append(f"| {case['case']} | {case['category']} | {', '.join(e['tags'])} | {e['evidence']} |")
    lines+=["","## 4. 对后续方案的约束","",
      "1. 语义条件值得重视：共同错误中包含人数/对象漏计、关系方向读反、总量与剩余量混淆、单位与计数粒度不符。很多错误算式本身完全算对，单纯提高算术检查权重无法覆盖这些问题。",
      "2. 算术错误只能作为软证据：第5、7、9例的正确候选也有错误中间推理，第9例甚至出现错误抵消。不能把有错误等式的候选直接淘汰，也不能把最终数值正确解释成整条推理正确。",
      "3. 不能将人工理解强行变为新真值：第18例按单只袜解释的180有合理性，但课程参考为90；第16例by December也存在累计口径歧义。保留官方标签和解析，不在后续特征中硬编码这些个案或改写评分标准。",
      "4. 有限树对照检验条件交互是否比现有模型更合适，完整参考配置检验特征、裁剪及正则化组合。它们都不自动解决语义理解；若没有稳定收益，应收束本轮，不能再用同一验证集无期限探索。","",
      "本阶段未使用验证题失败案例，也未查看测试内容。训练数据哈希与旧OOF来源哈希见summary.json，各种子分层结果亦保存在其中。原冻结资料和旧研究记录保持一致。"]
    (folder/"DIAGNOSIS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(folder/"case_review_provenance.json",{"cases_sha256":sha(folder/"cases.json"),"review_sha256":sha(folder/"CASE_REVIEW.json"),
         "case_count":len(cases),"stable_common_sample_multilabel_counts":dict(counts),"not_population_error_frequencies":True,
         "reviewer_is_AI":True,"no_new_model_fit_or_validation":True})
if __name__=="__main__":main()
