# V3-lite 因果增量验证协议

## 目标与边界

V3-lite 验证因果数据门禁、uplift learner 和离线 targeting policy，不把 Criteo 营销实验
解释成 Amazon 商品重排实验。Amazon Reviews 2023 缺少候选级曝光、对照、propensity、
logging policy 和固定转化窗口，因此只能用于 V1/V2 相关性推荐。

## 数据门禁

- Criteo 源文件必须通过长度、SHA-256、二值字段、随机处理组、overlap 和 ESS 审计；
- 正式实验使用从全部 13,979,592 行中抽取的 1,000,000 行 uniform reservoir sample，
  固定 sampling seed=2026，不使用前缀样本或合成样本；
- 80/10/10 deterministic split，split seed=1701；
- Validation 选择 learner 和策略权重；冻结 `selection.json` 后才运行 Test。

## 学习器与评测

- S-Learner、T-Learner、X-Learner、DR-Learner；
- 三折 out-of-fold nuisance cross-fitting；
- random targeting 与 response model 基线；
- AUUC、Qini/row、uplift@K、IPW policy value、每千样本估计增量转化；
- outcome Brier、log loss、ECE；
- seed 2026/2027/2028 Student-t 95% CI；
- Test 使用 200 次配对、按处理组稳定的 Poisson row bootstrap。

## 策略模拟

验证集从 `response_rank + lambda * uplift_rank` 中选择 `lambda`，要求 response proxy
保留率至少 97%。这只是 Criteo targeting policy，不报告 Recall/NDCG，也不称为 Amazon
商品级重排。

## 验收门

只有同时满足以下条件才报告“辅助 benchmark 上的稳定增量证据”：

1. 冻结策略 Test Qini/row bootstrap 95% CI 下界大于 0；
2. Test policy incremental@10% bootstrap 95% CI 下界大于 0；
3. 相对 random targeting 的 Qini/row 差值 CI 下界大于 0。
4. 相对 response-only 基线的 Qini/row 与 policy incremental@10% 配对 bootstrap
   差值 CI 下界均大于 0，才能把收益归因于 uplift 模块，而不是一般响应率排序。

无论是否通过，Amazon 因果主线仍保持 `not_identifiable`，直到提供真实候选级日志。
