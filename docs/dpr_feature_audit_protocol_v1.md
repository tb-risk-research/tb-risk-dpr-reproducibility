# 特征可用性与编码审计方案（2026-10-02）

本步不改变原始数据、生产代码或冻结主分析。逐项核对29个基线特征和4个SOP特征的原始变量、分类值、缺失、编码和时间可用条件，并输出正式队列特征表。

原论文报告家庭人口调查、问卷及检测；TST在放置后48–72小时读取。公开表缺少逐人筛查/结果返回日期，基线记录不等于逐项提前可用证明。num_contacts与原始每户记录人数完全相同；假定先完成人口调查再排序。死亡者可在家属同意下入组，因此死亡状态不必然是未来信息，但其具体记录时间未在表中保留。

## 结果生成前固定的两项RF敏感性

同一完整病例、结局、20种子、户级五折、RF参数；每情景比较基线与基线+4SOP；每臂复用折内BMI填补及prior_rate训练折回退。每项2,000次固定预测家庭bootstrap（seed91208）。不得按结果重新选择变量。

1. **保守信息可用情景**：两臂同时移除 idx_dead、hiv_pos_h、hiv_unknown_h、idx_smear_pos、idx_smear_known、idx_xpert_pos、idx_hiv_pos（7列）。保留人口调查、问卷及体格测量，并假定这些先于排序获取；不宣称这些已有真实时间戳。此情景为低信息假设，不证明部署合法性。
2. **编码修正情景**：父母关系明确匹配 Parent/Parent-in-law；contact HIV unknown包含原始空值；新增6个指示变量 diabetes_missing、airspace_missing、timespent_missing、relationship_missing、index_hiv_unknown、index_xpert_not_done。保留其他冻结编码（吸烟列实际指示never-smoking；BMI训练折中位、咳嗽固定30、时间缺失归中类）。此项量化明显漏编码及未知状态混并对结果的影响，不将修正后的较好结果自动升为主分析。

说明：新增未知指示保留“未知/未做”状态，不将其称为真实阴性，也不对缺失结局做填补。单个敏感性只支持该情景下的结论；不把编码修正或去检测变量实验称为全流程bootstrap、不将两个实验合并为未运行的共同修正情景。

来源：[HomeACF原论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230376)，本地raw RDA及 validation/real_data_infection.py，run_sop_unified_ci_leakfree.py。申请和临床流程验证不在本步执行范围。
