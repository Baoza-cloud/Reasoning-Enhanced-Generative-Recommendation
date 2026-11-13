# V2.7 候选级 Listwise 联合评分协议

状态：**已结束并冻结（阶段 0 通过，seed 2026 严格门禁失败，多种子未启动）**。

## 冻结结果

固定的 32 维 listwise joint scorer 已在全部 `339,519` 条 Train 样本上训练 20 epoch，使用
`510,520` 个排序对、`25,526` 个 Top-10 boundary 对和 `204,000` 个保护对。Test 未读取。

| Scale | Recall@10 | 保留率 | Tail Recall@10 | Tail 净命中 | Tail bootstrap 95% CI | Tail 覆盖增量 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.046031 | 0.9985 | 0.000273 | +1 | [-0.000182, 0.000454] | +0.037270 |
| 0.50 | 0.020752 | 0.4502 | 0.005635 | +60 | [0.004090, 0.006908] | +0.954494 |
| 0.75 | 0.001602 | 0.0347 | 0.008362 | +90 | [0.006453, 0.009910] | +0.967379 |
| 1.00 | 0.001602 | 0.0347 | 0.008362 | +90 | [0.006544, 0.009907] | +0.967379 |

scale 0.25 基本保护总体准确率并扩大覆盖，但尾部仅净增 1 个命中且区间跨 0；scale 0.50
以上能稳定增加尾部命中，却大量覆盖生成候选，scale 0.75/1.00 的推荐几乎 100% 为尾部，
总体 Recall 崩溃。因此没有 scale 同时通过准确率保护与尾部增益门禁，
`selected_validation_variant=null`，阶段 3 多种子按停止规则不运行。

V2.7 证明“去掉残差硬上限可以跨越 Top-10 边界”，同时表明当前 rank/semantic/frequency
特征无法可靠判断哪个检索候选才是真目标。瓶颈已从分数量级转移为用户—商品判别表示。
完整结果见 `artifacts/v2_7_listwise_joint/seed_2026/report.json`。

## 阶段 0：冻结 Validation 边界诊断

诊断只读取 seed 2026 的冻结 Validation 候选缓存，Test 未读取。Observed-Tail 基线
Recall@10 为 `0.000182`，Beam-20、BLaIR Tail-50 与候选并集 oracle 分别为
`0.000364/0.063534/0.063716`。并集中共有 `701` 个尾部真目标，其中 `697` 个是
BLaIR-only；`251` 个 BLaIR-only 目标的语义 rank 不超过 10。

V2.6 对这 `697` 个目标需要的 target-vs-cutoff pair lift 固定约为 `0.526316`，实际学习
到的中位 lift 为 `0.511432`，最终目标距 Top-10 边界的中位差仍为 `-0.014976`。虽然
`694/697` 在两侧残差理论极限下可达，但实际转入 Top-10 的数量为 `0`。scale-1 只在
`1,601/57,439` 行发生一次 Top-10 商品替换，整体新增/损失命中为 `7/10`，尾部为
`0/0`。因此 V2.6 的失败机制是硬边界与有界残差饱和，不是候选完全缺失。

阶段 0 工程门禁冻结为：union 尾部 oracle 相对基线至少增加 `0.01`，且至少存在 `100`
条 BLaIR-only 尾部真目标。两项均通过。该门禁只说明值得开发新 scorer，不构成模型增益
证据。结构化结果见 `artifacts/v2_7_top10_boundary/seed_2026/report.json`。

## 阶段 1：固定模型与损失

候选固定为 `Beam-20 ∪ BLaIR Observed-Tail-50`，不扩大 Beam，不使用强制配额。每个候选
使用以下真实特征：

- 生成排名/分数、BLaIR 语义排名/相似度、候选来源；
- Head/Torso/Observed-Tail 分组与 log-frequency；
- 冻结生成器给出的 Beam margin/entropy、历史长度、语义 gap 与检索候选占比；
- 语义、来源、频率的交互特征。

scorer 使用小型候选 MLP 输出不设人工双侧上限的校准 logit。训练目标由三部分组成：

1. 对真实目标与当前第 10 名及 hard negatives 的 listwise/boundary ranking loss；
2. 对原 V2 正确 Top-10 的 baseline-preservation loss；
3. 参数、logit shift 与行内分数漂移正则。

为避免在结果出来后追逐超参数，seed 2026 首轮固定为：hidden dim `32`、`tanh`、AdamW、
20 epochs、learning rate `0.003`、batch size `4096`、pair temperature `0.10`、Top-10
boundary margin `0.02`、tail positive weight `8`、boundary weight `4`、preservation weight
`4`、distillation weight `0.01`。只评估预先列出的 residual scale
`0.25/0.50/0.75/1.00`，不追加网格；若全部失败则停止本轮。

不得使用 Test 调参，不得用 soft quota 或手工把尾部商品塞入 Top-10。所有训练仍来自真实
Train；Validation 仅用于 seed 2026 门禁。

## 阶段 2：seed 2026 严格门禁

- 总体 Recall@10 保留率至少 `0.99`；
- NDCG@10 增量不低于 `-0.0002`；
- Observed-Tail Recall@10 净新增至少 `6` 个命中；
- 尾部用户 bootstrap 95% CI 下界大于 `0`；
- Observed-Tail 目录覆盖增量至少 `0.02`；
- 同次运行新增评分延迟不超过 `5 ms/example`。

任一门禁失败即冻结 V2.7 负结果，不追加种子。

## 阶段 3：多种子与外部确认

仅当阶段 2 全部通过，才运行 `2026/2027/2028/2029/2030` 五种子。至少 4/5 个种子方向
为正，并同时报告 Student-t、exact McNemar、用户 paired bootstrap、层次 bootstrap 与
Holm 校正。模型和阈值冻结后原样迁移 Sports/Toys；外部品类不得重新搜索权重。Test 只在
全部结构冻结后评估一次。

## 当前科学边界

阶段 0 证明的是“候选集合存在很大的 oracle 空间，且旧 scorer 卡在可量化的 Top-10
边界”，而不是“V2.7 已有增益”。seed 2026 随后证实新 scorer 存在无法通过门禁的
准确率—尾部增益冲突，因此 V2.7 不升级；默认方案仍是原 V2
`token_loss_010 + Beam-20/Top-10`。
