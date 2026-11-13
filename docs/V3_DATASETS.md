# V3 因果数据选择

## 结论

V3 使用两类真实数据，各自回答不同问题：

1. [Criteo Uplift](https://huggingface.co/datasets/criteo/criteo-uplift) 验证“营销处理是否带来
   增量转化”。官方数据来自随机 incrementality tests，约 1,398 万行，包含 12 个特征、
   `treatment`、`conversion`、`visit`、`exposure`。这是 uplift learner 主验证集。
2. [Open Bandit Dataset](https://github.com/st-tech/zr-obp) 验证“商品 action 重排策略能否用
   logged propensity 做 OPE”。它来自 ZOZOTOWN 时尚电商，含 item action、click reward、
   position 与真实 `pscore`。这是候选重排/OPE 验证集，不等价于转化 uplift RCT。

[KuaiRand](https://github.com/chongminggao/KuaiRand) 含随机曝光和多反馈，适合曝光偏差、序列
推荐和 OPE 扩展；它是视频域，暂列外部验证，不替代 Criteo 的 conversion uplift。

## 不使用 Amazon Reviews 2023 做 V3 因果结论的原因

Amazon Reviews 2023 没有候选级 treatment/control、统一归因窗口 conversion 和 logging
propensity。评论只在用户已经采取行为后才出现，选择机制本身不可忽略。把评论存在当处理、
把未评论商品当未转化都会制造标签和反事实。

## Criteo 使用限制

官方说明数据经过非均匀下采样，无法恢复原始总体的绝对 incrementality 水平。因此本项目用
它比较 uplift 排序和策略价值，不报告总体 ATE。下载和全量审计：

```bash
python scripts/prepare_criteo_uplift.py --download
```

全量审计结果为 13,979,592 行、11,882,655 treatment、2,096,937 control 和 40,774
conversion；处理概率为 0.850000，最小处理臂概率为 0.150000，IPW ESS 约 7,129,587，
overlap 门通过。源文件 SHA-256：
`2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`。

## V3-lite 冻结结论

正式实验从全部已审计行中 uniform reservoir 抽取 1,000,000 行真实样本，使用固定 sampling
seed 2026、80/10/10 切分、三训练种子和三折 nuisance cross-fitting。Validation 选择
DR-Learner 和 `response_rank + 0.5 * uplift_rank`，冻结后再评估 Test。

冻结策略在 Test 上的 Qini/row 为 0.0004235，bootstrap 95% CI 为
`[0.0001197, 0.0007030]`；policy incremental@10% 为 0.0012437，区间为
`[0.0005855, 0.0018863]`。相对 random targeting 的差值稳定为正，但相对 response-only
基线的 Qini/row 差值区间 `[-0.0000247, 0.0000286]`、policy incremental 差值区间
`[-0.0000121, 0.0001644]` 均跨 0。因此 targeting 信号成立，uplift 模块的额外稳定增益
不成立。详细边界见 [V3_FINAL_REPORT.md](V3_FINAL_REPORT.md)。
