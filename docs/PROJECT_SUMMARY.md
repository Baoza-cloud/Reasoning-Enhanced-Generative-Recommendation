# 项目摘要与面试讲解

## 一句话介绍

基于 Amazon Reviews 2023 构建端到端生成式推荐系统：用 BLaIR 与 RQ-VAE 将商品映射为
离散 Semantic ID，通过行为感知 Transformer 生成候选商品，并围绕长尾推荐、偏好融合和
因果增量评估建立多种子、可复现的实验体系。

## 60 秒项目介绍

传统序列推荐通常对完整商品表打分，商品规模增长后输出层和检索成本都会上升。本项目先将
商品文本向量压缩为三层 Semantic ID，再把推荐转化为受 Catalog Trie 约束的自回归生成。
用户侧同时编码行为类型、评分强度和时间间隔，并融合长期偏好与近期意图。

项目主数据使用 Amazon Reviews 2023 Musical Instruments，包含 339,519 条训练样本、
57,439 条验证样本、57,439 条测试样本和 24,587 个商品。正式实验使用三个独立训练种子，
报告 Recall/NDCG、目录覆盖、长尾命中、延迟和置信区间。最终默认方案的 Recall@10 为
4.0019%，与 SASRec 处于同一量级；温和长尾重排将总目录覆盖从 2.62% 提升到 8.48%，但
尾部命中增量区间跨 0，因此没有将其升级为默认方案。

因果部分另外在 Criteo 随机实验数据上完成候选资格审计、S/T/X/DR Learner、cross-fitting
和 bootstrap。策略稳定优于随机排序，但没有稳定优于 response-only 强基线，因此最终只
保留工程框架与 targeting 结论，不声称 Amazon 商品级增量转化收益。

## 简历项目描述

### 推荐系统方向

- 基于 24,587 个真实商品构建 `BLaIR → RQ-VAE → Semantic ID → Trie Beam Search` 生成式
  推荐链路；引入 K-means/EMA 码本训练与逐层利用率、perplexity、死码和碰撞诊断，三层
  码本利用率达到 90.63%/98.44%/100%，生成结果 valid-item rate 为 100%。
- 设计行为感知序列编码器，联合建模评分强度、行为极性、真实时间间隔及长期/短期偏好；
  在 Amazon Reviews 2023 完成 2026/2027/2028 三种子 Beam-20 全量评测，默认方案
  Recall@10 为 4.0019%，并与 SASRec、Popularity、Markov 和无强度版本统一对照。
- 建立 Head/Torso/Observed-Tail/Cold-start 分层评测和候选漏斗，联合分析 Recall/NDCG、
  目录覆盖、尾部覆盖与 CPU 延迟；温和重排将目录覆盖从 2.62% 提升到 8.48%，同时保留
  99.45% 的 Recall@10，并通过错误案例定位候选缺失与跨来源分数失配问题。

### 因果评估方向

- 实现候选级因果数据门禁、overlap/ESS、AIPW/DR pseudo-outcome、S/T/X/DR Learner、
  三折 nuisance cross-fitting 与 IPW policy evaluation；对 13,979,592 行 Criteo 数据完成
  全量审计，并在 100 万行正式样本上执行三种子与 200 次分层 bootstrap。
- 坚持 Validation-only 选择和 Test 冻结协议：联合策略相对 random targeting 的
  Qini/row 差值区间为 `[0.0001873, 0.0008476]`，但相对 response-only 的区间跨 0，
  因而拒绝把一般响应率排序收益解释为稳定 uplift 增益。

## 关键技术决策

### 为什么使用 Semantic ID

- 将大规模商品分类问题转化为短 token 序列生成；
- 语义相近商品能够共享前缀，降低完全独立商品 ID 的稀疏性；
- 能使用前缀树约束解码，保证每条生成路径都对应目录中的真实商品。

### 如何处理码本塌缩

- 使用分层 K-means 初始化，而不是完全随机初始化；
- 使用 EMA 更新 codebook，减少小批次梯度震荡；
- 监控 used/dead codes、usage ratio、perplexity、top-code share；
- 为碰撞商品添加稳定消歧 token，确保 Semantic ID 与真实商品可逆映射。

### 为什么最终选择 Beam-20

Beam-20 相比 Beam-10 的 Recall@10 提升 12.70%，但耗时增加 94.94%，目录覆盖下降
21.24%。项目将其定义为“准确率优先”的离线默认配置，而不是无条件认为更大的 Beam 更好。

### 长尾问题的真实瓶颈

36,216 个尾部目标中：

- 生成 Beam-20 只覆盖 15 个；
- BLaIR Tail@50 覆盖 1,860 个；
- 两路并集覆盖 1,867 个；
- 最终 Top-10 命中 14 个。

因此长尾失败分为两层：94.84% 的目标根本不在候选集中；候选已包含目标时，进入 Top-10
的转化率仍只有 0.75%。后续改进应优先统一生成与语义检索的用户—商品评分空间，而不是
继续提高强制配额。

### 为什么 V3 不直接使用 Amazon Reviews

评论数据只记录已经发生的行为，不包含请求级候选集合、未曝光但合格的对照商品、logging
policy、propensity 和固定转化窗口。把未评论商品直接视为未转化会构造错误反事实。因此
Amazon 因果门禁保持 `not_identifiable`，随机实验辅助验证使用 Criteo。

## 结果边界

- 默认生成方案与 SASRec 同量级，但没有统计证据证明稳定优于 SASRec；
- 行为强度版本均值略高，但配对置信区间跨 0；
- 温和长尾重排稳定提高覆盖率，但尾部 Recall 增量未达到稳定门槛；
- Preference Reasoning 中近期事件遮蔽有稳定影响，但组件独立重训差值区间跨 0；
- V3 targeting 策略稳定优于随机，但 uplift 模块未稳定优于 response-only；
- Beauty、Toys、Sports 只完成真实数据和商品向量准备，不计作效果验证；
- UniSRec 因本地资源限制停止，没有正式结果。

## 常见追问

### 为什么 Recall@10 只有约 4%？

任务是在 24,587 个真实商品中预测 leave-last-out 的单个下一商品，并且没有使用曝光、搜索、
价格、店铺、库存等线上上下文。项目重点是完整生成链路和严格同口径对照，不把不同数据切分
或候选范围下的指标直接横向比较。

### 为什么保留大量负结果？

推荐改动往往会同时影响准确率、覆盖率和延迟。只报告单 seed 或单指标很容易把随机波动解释
为收益。项目把未通过门槛的方案冻结下来，能够说明实验设计、停止规则和结果边界。

### 如果继续迭代，最优先做什么？

1. 学习候选级用户—商品联合 scorer，统一生成候选与语义检索候选的分数量级；
2. 接入真实曝光、候选集合、position、propensity 与固定转化窗口日志；
3. 增加在线推理服务和 P50/P95/P99、QPS、内存压测；
4. 在资源允许时完成更多品类和现代序列推荐基线。

## 证据索引

- 推荐主报告：[V2_FINAL_REPORT.md](V2_FINAL_REPORT.md)
- 因果辅助报告：[V3_FINAL_REPORT.md](V3_FINAL_REPORT.md)
- 全部实验记录：[../EXPERIMENTS.md](../EXPERIMENTS.md)
- 默认配置：[../configs/v2_final_default.yaml](../configs/v2_final_default.yaml)
- 部署模式：[../configs/v2_final_modes.yaml](../configs/v2_final_modes.yaml)
- V3-lite 配置：[../configs/v3_lite.yaml](../configs/v3_lite.yaml)
