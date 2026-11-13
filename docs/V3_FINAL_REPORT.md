# V3-lite 最终报告

## 最终结论

V3-lite 的工程闭环已经完成：候选级因果资格门禁、真实 RCT 数据审计、overlap/ESS、
S/T/X/DR learner、三折 nuisance cross-fitting、Validation-only 策略选择、三种子
Student-t CI、Test 配对 bootstrap 和 targeting policy 模拟均可复现。

算法结论必须分成两层：

1. 冻结策略在 Criteo 上稳定优于 random targeting，说明特征中存在可用于 targeting 的
   排序信号；
2. 冻结策略相对 response-only 强基线的增量区间跨 0，不能证明 uplift learner 带来
   额外稳定收益，也不能外推成 Amazon 商品级增量转化提升。

因此 V3-lite 状态冻结为
`completed_without_stable_incremental_uplift_over_response`。

## 数据真实性与因果资格

### Criteo Uplift

- 源数据：`criteo/criteo-uplift`；
- 全量审计：13,979,592 行；
- treatment/control：11,882,655 / 2,096,937；
- conversion：40,774；
- 处理概率：0.850000；最小处理臂概率：0.150000；
- IPW ESS：约 7,129,587，overlap@0.01 通过；
- 源文件 SHA-256：
  `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`；
- 正式实验：从全部已审计行中 uniform reservoir 抽取 1,000,000 行，sampling seed=2026；
- Train/Validation/Test：800,003 / 99,995 / 100,002，split seed=1701；
- `synthetic_data_used=false`。

Criteo 官方数据经过非均匀下采样，本文只解释排序和样本尺度 policy 指标，不把 ATE 或
“每千样本增量”解释为原始总体的绝对转化提升。

### Amazon Reviews 2023

候选级门禁结果为 `not_identifiable`。缺失字段包括 request-level candidate set、eligible
control、treatment/exposure、outcome、conversion time、position、logging policy、
propensity 与固定转化窗口。Amazon 仍只用于 V1/V2 相关性推荐。

## 冻结实验协议

- learner：S-Learner、T-Learner、X-Learner、DR-Learner；
- baseline：random targeting、response-only model；
- nuisance：三折 out-of-fold cross-fitting；
- 模型种子：2026、2027、2028；
- Validation 主指标：三种子 mean Qini/row；
- 策略：`response_rank + lambda * uplift_rank`，response proxy 保留率至少 97%；
- Validation 冻结结果：DR-Learner、`lambda=0.5`；
- Test：冻结 `selection.json` 后运行；
- 区间：三种子 Student-t 95% CI、200 次配对分层 Poisson row bootstrap。

## 主要结果

### 各学习器 Test 三种子均值

| Scorer | Qini/row | Policy incremental@10% |
|---|---:|---:|
| Random | -0.0000751 | 0.0000264 |
| Response-only | 0.0004198 | 0.0011810 |
| S-Learner | 0.0004199 | 0.0011285 |
| T-Learner | 0.0003800 | 0.0011304 |
| X-Learner | 0.0000081 | 0.0005158 |
| DR-Learner | 0.0004398 | 0.0013462 |

DR-Learner 的三种子 Qini/row 95% Student-t CI 为
`[0.0004160, 0.0004635]`；policy incremental@10% 区间为
`[0.0012854, 0.0014070]`。这些绝对区间说明排序方向稳定，但不能替代强基线配对比较。

### 冻结联合策略 Test

| 指标 | 点估计 | Bootstrap 95% CI |
|---|---:|---:|
| Qini/row | 0.0004235 | [0.0001197, 0.0007030] |
| Policy incremental@10% | 0.0012437 | [0.0005855, 0.0018863] |
| 每千样本增量（下采样尺度） | 1.2437 | [0.5855, 1.8863] |

### 配对强基线检验

| 对照 | Qini/row 差值 95% CI | Policy incremental@10% 差值 95% CI | 结论 |
|---|---:|---:|---|
| Random targeting | [0.0001873, 0.0008476] | [0.0007496, 0.0017836] | 稳定更好 |
| Response-only | [-0.0000247, 0.0000286] | [-0.0000121, 0.0001644] | 区间跨 0 |

三种子配对 Student-t 结果与 bootstrap 一致：联合策略相对 response-only 的 Qini/row
差值区间为 `[-0.0000002, 0.0000108]`，policy incremental 差值区间为
`[-0.0000411, 0.0002033]`。

## 校准与限制

T-Learner 和 S-Learner 均报告 factual Brier、log loss 与 ECE。seed 2026 Test 的实际
转化率为 0.002800，而 S/T factual mean probability 分别为 0.001151/0.001245，存在明显
低估；因此模型概率不能直接当作已校准的线上绝对转化率。当前 policy value 来自 RCT
holdout 的 IPW 评估，而不是把模型概率直接相加。

Criteo 没有 Amazon 用户商品候选集合和相关性标签，不能报告 Recall/NDCG，也不能完成
Amazon 的 `relevance + lambda * uplift` 商品级线上重排验证。

## 复现

```bash
PYTHONPATH=src python scripts/audit_causal_readiness.py
PYTHONPATH=src python scripts/prepare_criteo_uplift.py
PYTHONPATH=src python scripts/run_v3_lite_benchmark.py --config configs/v3_lite.yaml
```

主要产物：

- `artifacts/v3_lite_formal/report.json`：
  `0bd2824d326964610d2c927f8db7d33445f5f49c23c009355b7af2b7a8a3a1b0`；
- `artifacts/v3_lite_formal/selection.json`：
  `8880c33ca96825899cd5cc1d16adf17788fd4f5f4dd000fd70827cc0e790135a`；
- `artifacts/v3_lite_formal/official_test_scores.npz`：
  `81f2a8540183cc5dad4894ed74a671340deb095571e144c9415c30007cbaa253`；
- `artifacts/v3_lite_formal/sample.pt`：
  `c1ed89ab2a276602a19d9ce7d80b54b658531a6104169ad13416ed541bbe1bd8`。

## 简历表述边界

可以表述：实现候选级因果数据门禁、cross-fitted S/T/X/DR learner、IPW policy evaluation
及 bootstrap 统计框架，并在百万级真实 RCT 样本上完成验证集冻结和强基线审计。

不应表述：Amazon 推荐获得稳定增量转化提升；或 uplift learner 已稳定优于 response-only
模型。
