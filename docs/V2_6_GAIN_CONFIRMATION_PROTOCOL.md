# V2.6 增益确认预注册协议

状态：**已结束并冻结（阶段 1 通过，阶段 2 失败，阶段 3 未启动）**。

## 冻结结果

阶段 1 对全部 `339,519` 条真实 Train 样本完成可恢复分片普查，得到 `3,310` 条
Observed-Tail hard positives，覆盖 `2,838` 个用户与 `2,611` 个目标商品。三个数据门禁
全部通过；`synthetic_data_used=false`，Test 未读取。

阶段 2 使用全量 hard positives 训练 seed 2026。Validation 冻结基线 Recall@10 为
`0.046101`，Observed-Tail Recall@10 为 `0.000182`。部署 scale `0.25/0.50/0.75` 没有
产生尾部命中变化；scale `1.0` 的 Recall@10 为 `0.046049`，尾部 Recall@10 仍为
`0.000182`，尾部新增/损失命中为 `0/0`，尾部覆盖只增加 `0.004404`，低于 `0.02`
门槛。故没有方案通过阶段 2，阶段 3 五种子与外部确认按预注册停止规则未启动。

该结果排除了“仅因 Train hard-positive 数量不足”的解释，但不能证明所有联合 scorer 都
无效。后续 V2.7 只针对已确认的 Top-10 边界问题重新设计；原 V2 继续为默认方案。

## 唯一主张

V2.6 只检验一个主张：候选感知的语义—生成联合学习能在基本保持总体准确率的前提下，
稳定提高 Observed-Tail Top-10 命中。Preference Reasoning 与因果 uplift 不属于本轮主要
增益主张。

## 数据隔离

- Musical Instruments 只使用 Train 拟合、Validation 选择和停机；不得读取 Test。
- Head/Torso/Observed-Tail/Cold-start 固定为训练频率 Top 20% / 中间 30% /
  已出现商品 Bottom 50% / 训练频率为 0，不得事后修改。
- 所有结构、超参数和停机规则冻结后，最终确认优先使用未参与开发的 Sports and Outdoors
  与 Toys and Games。
- 主线只接受 `synthetic_data_used=false` 且数据审计通过的产物。

## 阶段 1：全量 Train hard-positive census

对全部 Train 滚动 next-item 样本计算 `Beam-20 ∪ BLaIR Observed-Tail-50`。hard positive
定义为真实目标位于 BLaIR 候选、但不位于 Beam-20。普查必须分片、可恢复，并报告：

- 总样本、Beam/BLaIR/union oracle；
- hard-positive 行数、唯一用户数、唯一商品数；
- Head/Torso/Observed-Tail/Cold-start 分组；
- BLaIR rank、目标训练频率和历史长度直方图；
- 所有输入、分片和汇总产物的 SHA-256；
- `test_evaluated=false`。

阶段 1 go/no-go 门槛：至少 `500` 个 Observed-Tail hard-positive 行、至少 `100` 个唯一
Observed-Tail 商品、至少 `500` 个唯一用户。任一门槛失败则停止学习型长尾 scorer。

## 阶段 2：seed 2026 Validation 门禁

仅当阶段 1 通过时训练一个零初始化、有界残差的候选感知模型。原生成分数作为不可覆盖
主干；BLaIR 分数按频率层校准；训练使用真实目标、Beam/BLaIR hard negatives、pairwise
或 listwise loss 与 baseline-preservation loss。

seed 2026 进入多种子的必要条件：

- 总体 Recall@10 保留率至少 `99%`；
- NDCG@10 增量不低于 `-0.0002`；
- Observed-Tail Recall@10 至少净新增 `6` 个命中，且用户 bootstrap 95% CI 下界大于 0；
- Observed-Tail 目录覆盖增加至少 `0.02`；
- 同次运行新增延迟不超过 `5 ms/example`。

## 阶段 3：五种子与外部确认

seed 固定为 `2026/2027/2028/2029/2030`。至少 4/5 个种子主要指标方向为正，且种子级
Student-t、用户级 paired bootstrap、exact McNemar 与层次 bootstrap 的结论一致；多方案
比较使用 Holm 校正。若五种子方向仍混乱，不得通过继续追加种子追逐显著性。

冻结配置随后原样迁移到 Sports/Toys；测试集一次性评估，不重新搜索权重。只有外部品类
方向一致且至少一个品类的主指标 95% CI 下界大于 0，才表述为可泛化的稳定增益。

## 停止规则

- 不扩大 Beam 代替学习信号修复；
- 不使用软配额制造命中；
- 不根据 Test 结果选参或翻转停止决定；
- 不使用相对百分比掩盖少量离散命中；
- 任一阶段失败即保留原 V2 默认方案，并如实冻结负结果。

以上停止规则已执行：`selected_validation_variant=null`，没有追加随机种子，也没有读取
Test 追逐显著性。完整结果见：

- `artifacts/v2_6_hard_positive_census/full_seed_2026/report.json`
- `artifacts/v2_6_full_hard_positive/seed_2026/report.json`
