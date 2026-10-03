# 特征可用性、编码与正式样本特征表：本轮总结

本步已逐项核对29个基线预测特征、4个SOP特征；从原始数据整理正式Table 1和补充Table S1，并完成两个20种子RF敏感性情景（合计400次RF拟合、每情景2,000次条件家庭重抽）。160项AUROC/AUPRC重算及数据/方案/脚本哈希核对通过。

## 发现与处理

- 父母关系编码漏匹配：原始标签 Parent/Parent-in-law 在全部数据中有 345 人，分析人群有 318 人；冻结 rel_parent 全部为零。已在独立敏感性修正，不隐藏旧主分析的错误。
- smoke_ever_h实际是“从未吸烟”指示，不是“曾经吸烟”。这是名称与解释错误；本步修正文稿解释，不翻转既有数值。
- contact HIV显式unknown与空值不是一回事；旧编码把3个空值并入非阳性/非unknown。修正情景将空值并入unknown，并添加六个未知/未做状态指示。
- 涂片未做、Xpert未做、HIV未知属于有意义的类别，不等于数据库空值，也不等于阴性。正式表分别呈现，不能把0%空值写成检测全部完整。
- num_contacts逐行等于原始完整家庭名单人数，包含没有TST结果的成员；它不是按完整病例筛掉成员后的户大小。可用性前提是先完成家庭人口调查。

## 两项敏感性结果

| 情景 | 基线AUROC | 加SOP AUROC | 平均增益 | 条件95%区间 | 正向种子 |
|---|---:|---:|---:|---|---:|
| 去除7个检测/状态特征 | 0.6478 | 0.7158 | +0.0680 | [+0.0462, +0.0898] | 20/20 |
| 修正父母关系及未知编码 | 0.6433 | 0.7126 | +0.0693 | [+0.0469, +0.0903] | 20/20 |

这是两个单独情景，不能假称完成了两者一起修正的实验；区间条件于拟合后的预测，不是完整重拟合区间。它们不替代冻结主分析。

## 时间可用性可以确认到什么程度

[原论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230376)报告基线家访包含家庭人口调查与问卷、向接触者提供HIV检测，TST放置后48–72小时读取，且允许死亡指示病例经家属同意入组。因此死亡变量不必然是未来标签；但公开表没有逐人记录时间来验证可用顺序。
年龄/性别/关系/既往暴露可被设为人口调查后的排序输入；BMI要求先做体格测量；HIV和指示病例检测结果要求已获知；SOP要求先前TST读数真正返回。后者在本稿只有模拟条件，不能改成“真实部署信息全部合法”。没有筛查日期、结果返回日期或逐次行动记录，这一事实无法靠重新跑模型修复。

## Supplementary Table S2. Frozen predictor specification and availability

空值计数顺序为原始2,985人/有TST的2,725人；未知和未做类别见正式补表。

| 建模特征 | 原始变量 | 冻结编码/缺失处理 | 可用性条件 | 原始空值：全部/完整病例 |
|---|---|---|---|---:|
| contact_age | ageyears_h | Age, numeric | Questionnaire/census assumed available; no individual timestamp | 0/0 |
| age_lt5 | ageyears_h | Age <5 years | Derived from age | 0/0 |
| age_ge45 | ageyears_h | Age >=45 years | Derived from age | 0/0 |
| contact_sex_m | sex_h | Male=1; Female=0 | Census assumed available | 0/0 |
| hiv_pos_h | hivfinal_h | HIV positive=1; unknown/empty not positive | Test/history result timing not separable | 3/3 |
| hiv_unknown_h | hivfinal_h | Explicit HIV unknown=1; raw empty=0 in frozen coding | Test/history timing not separable | 3/3 |
| bmi_h | bmi_h | Numeric; restore raw missing; training-fold median | Anthropometry must precede ranking; cost not measured | 17/16 |
| smoke_ever_h | smoke_h | Actually never-smoking=1; current/previous=0 | Questionnaire assumed available; legacy name misleading | 0/0 |
| diabetes_h_f | diabetes_h | Yes=1; No and raw empty=0 | History question; empty differs from negative | 3/1 |
| site_capricorn | site | Capricorn=1; Mangaung=0 | Study-site context | 0/0 |
| hh_n_contacts | num_contacts | Original roster size, not complete-case household size | Full household roster assumed known before ranking | 0/0 |
| idx_coughdays | coughdays_i | Numeric; raw missing fixed at 30 days | Index history; individual collection time absent | 9/7 |
| idx_smear_pos | smear_i | Smear positive=1; negative/not done=0 | Index laboratory availability time absent | 0/0 |
| idx_smear_known | smear_i | Positive or negative=1; not done=0 | Explicit test-not-done distinction | 0/0 |
| idx_xpert_pos | xpert_i | Xpert positive=1; negative/not done=0 | Index laboratory availability time absent | 0/0 |
| idx_hiv_pos | hiv_i | HIV positive=1; negative/unknown=0 | Index history/test timing absent | 0/0 |
| idx_age | ageyears_i | Numeric age | Index registration assumed available | 0/0 |
| idx_sex_m | sex_i | Male=1; Female=0 | Index registration assumed available | 0/0 |
| idx_dead | dead_i | Deceased=1; not deceased=0 | Deceased index cases eligible in source study; timing per row absent | 0/0 |
| ts_low | timespent_h | a) every now and again=1 | Exposure history; recalled | 3/2 |
| ts_mid | timespent_h | Neither low nor high=1; missing assigned middle | Exposure history; raw missing neutralized | 3/2 |
| ts_high | timespent_h | c) most of the day=1 | Exposure history; recalled | 3/2 |
| share_bedroom | sharebedroom_h | Yes=1; No=0 | Exposure history | 0/0 |
| sleep_same_bed | sleepsamebed_h | Yes=1; No=0 | Exposure history | 0/0 |
| airspace_shared | airspace_h | Yes=1; No and raw empty=0 | Exposure history; empty differs from No | 1/1 |
| rel_child | relationship_h | Exact Child=1 | Relationship from census | 1/1 |
| rel_spouse | relationship_h | Exact Spouse=1 | Relationship from census | 1/1 |
| rel_sibling | relationship_h | Exact Brother/sister=1 | Relationship from census | 1/1 |
| rel_parent | relationship_h | Frozen match Parent; actual label Parent/Parent-in-law | Coding mismatch: frozen column constant zero | 1/1 |
| prior_n | earlier same-household TST | Count of results assumed available earlier | Simulated; observed return times absent | Structural zero for first member |
| prior_pos | earlier same-household TST | Positive count excluding target | Same availability constraint | Structural zero for first member |
| prior_rate | prior_pos/prior_n | Training-fold prevalence fallback when prior_n=0 | Same availability constraint | Undefined rate replaced within fold |
| prior_screened | prior_n | Indicator prior_n>0 | Same availability constraint | Structural zero for first member |

主模型明确禁止把目标成员自己的tstdiam_h、tst_pos10、tstdone_h或tstread_h放入基线预测列。已知先前成员结局在同户测试折内可作为模拟历史信息，但不跨户折进入训练。

## 投稿时的处理

正文保留冻结主结果并披露编码错配，展示修正敏感性；不把较好结果自动换成主结果。真实部署和所有变量都先于排序可得的主张继续禁止。正式Table 1区分“有TST”和“无TST”，HIV未知/空值分开；户数在两组重叠，不能相加；指示病例百分比明确按接触者加权。

## 下一步

统一主要图、正式表号和补充材料；把这些审计映射到TRIPOD+AI及复现清单。作者声明、伦理准确措辞、许可和归档信息仍需核定。本步不提交或推送。
