# V2.4 增强实验协议与最终结论

状态：**已结束并冻结（2026-11-10）**。V2.3 核心证据未被覆盖；V2.4 先完成
Validation-only 增强决策，随后按冻结配置完成一次性 Test 确认。最终没有任何 V2.4 方案
通过升级门槛，原 V2
`token_loss_010 + Beam-20/Top-10` 继续作为最终默认方案。

## 预注册问题与验收门槛

V2.4 检验四个问题：个性化可学习遗忘、生成—BLaIR 联合 scorer、Beauty/Toys/Sports
跨品类迁移，以及官方 UniSRec 现代基线。模型选择不得使用 Test；联合 scorer 同时要求：

- 总体 Recall@10 至少保留生成基线的 97%；
- Observed-Tail Recall@10 的三种子配对 Student-t 95% CI 下界大于 0。

种子固定为 `2026/2027/2028`。动态遗忘与联合 scorer 先只用 Musical Instruments
Validation 决策；在配置、checkpoint、scorer 和“不升级”结论冻结后，Test 仅执行一次，
不参与训练、选参或重选。

## 已冻结结论

### 1. 个性化动态遗忘：未升级

配置为 `history_forgetting_mode=learned_exponential`，其余保持 3×256 RQ-VAE、EMA 0.99、
dim 128、2 层 encoder/decoder、10 epoch、token-loss alpha 0.10 和 Beam-20/Top-10。
下表为三种子 Validation 均值，区间是 adaptive−frozen V2 的配对 Student-t 95% CI：

| 指标 | Frozen V2 | 动态遗忘 | 配对增量 95% CI | 结论 |
|---|---:|---:|---:|---|
| Recall@5 | 0.028215 | 0.027351 | [-0.001449, -0.000281] | 稳定下降 |
| NDCG@5 | 0.018065 | 0.017948 | [-0.000945, +0.000712] | 无稳定差异 |
| Recall@10 | 0.044436 | 0.043043 | [-0.005261, +0.002476] | 未稳定提升 |
| NDCG@10 | 0.023274 | 0.022996 | [-0.000985, +0.000427] | 无稳定差异 |
| Observed-Tail Recall@10 | 0.000288 | 0.000227 | [-0.000321, +0.000200] | 无稳定提升 |

Recall@10 保留率约 96.87%，且增量区间跨 0。该机制不进入默认方案。

### 2. Learned hybrid scorer：正式否决

候选为 Beam-20 生成候选与 BLaIR Observed-Tail@50 的并集；Validation 用户按稳定哈希
固定拆成 70% fit / 30% selection。三种子 selection fold 结果如下：

| 指标 | Generator | Learned hybrid | 配对增量 95% CI |
|---|---:|---:|---:|
| Recall@5 | 0.028172 | 0.012079 | [-0.021363, -0.010822] |
| NDCG@5 | 0.018570 | 0.007059 | — |
| Recall@10 | 0.043916 | 0.025070 | [-0.022677, -0.015015] |
| NDCG@10 | 0.023635 | 0.011183 | — |
| Observed-Tail Recall@10 | 0.000104 | 0.000000 | [-0.000551, +0.000343] |

其 Recall@10 只保留生成基线的 **57.1%**，远低于 97% 门槛；尾部增量也未通过。
因此 learned hybrid scorer 被正式否决，只保留为候选漏斗诊断，不得作为部署方案。

### 2.1 冻结后 Test 确认

Test 为完整 57,439 条真实样本，固定 Beam-20、生成候选 20、BLaIR Observed-Tail 检索
50、Top-K 10。三种子均值如下：

| 指标 | 原 V2 默认 | 动态遗忘 | Learned hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.025186 | 0.024049 | 0.011380 |
| NDCG@5 | 0.016166 | 0.015624 | 0.006684 |
| Recall@10 | 0.040019 | 0.038452 | 0.022685 |
| NDCG@10 | 0.020927 | 0.020247 | 0.010280 |
| Observed-Tail Recall@10 | 0.000166 | 0.000110 | 0.000055 |
| 总目录覆盖 | 0.026166 | 0.025311 | 0.037540 |

动态遗忘的 Recall@10 保留率为 96.08%，相对原 V2 增量 95% CI
`[-0.005463, +0.002329]`；learned hybrid 保留率为 56.69%，增量 CI
`[-0.022460, -0.012208]`。McNemar 与 5,000 次用户 bootstrap 进一步确认 hybrid 在三个
种子上总体 Top-10 均显著下降。Test 因而确认而非改变 Validation 阶段的否决结论。

延迟仅在同次运行内比较：hybrid 相对动态遗忘增加约 3.284 ms/example。原 V2 与 V2.4
来自不同机器负载和并行运行，绝对延迟不作机制归因。

### 3. UniSRec：主动停止，无正式结果

适配器使用 `RUCAIBox/UniSRec` commit
`05aa5cba2809112c32808f70d16abc61c05c6538`、RecBole 1.2.1、本项目 Amazon 2023 固定
切分与 BLaIR 向量。CPU 单 worker 训练在本地资源下耗时不可接受，用户明确要求停止。
`2026/2027/2028` 均未形成完整训练和 full-sort 评测，故不报告、也不推断任何 UniSRec
对比结果。停止状态见 `artifacts/unisrec_amazon2023_musical_multiseed/state.json`。

### 4. 外部品类：真实数据和 BLaIR 已完成，模型验证未执行

All Beauty、Toys and Games、Sports and Outdoors 均来自
`McAuley-Lab/Amazon-Reviews-2023` 真实 5-core leave-last-out 数据；审计均为 `passed`，
`synthetic_data_used=false`。数据准备、引用一致性、时间顺序和 BLaIR 编码完成，但外部
V2/adaptive 三种子训练随资源队列主动停止，不能声称跨品类效果已验证。

| 品类 | Items | Train | Validation | Test | BLaIR SHA-256 |
|---|---:|---:|---:|---:|---|
| All Beauty | 356 | 1,776 | 253 | 253 | `494d4aba98781ceb7d092673b04b1975e91fa7caa6f30351b9e61d48ef3a68e8` |
| Toys and Games | 162,035 | 2,565,094 | 432,264 | 432,264 | `6ac0c72d795c1860c1543804db7c00b6ce90afe94f31a7d3836123c820a1313f` |
| Sports and Outdoors | 156,235 | 2,242,704 | 409,772 | 409,772 | `58ef22ef51c1f04e6609b9b6c1041a013dd1078340247dfa7ba93d39e3a80171` |

## 数据 SHA-256

以下哈希冻结官方 benchmark 原始四文件；逐个 processed JSONL、行数、字节数与商品引用
哈希保存在各品类的 `data_audit.json`，冻结清单对这些审计文件再次取 SHA-256。

| 品类 | 文件 | SHA-256 |
|---|---|---|
| All Beauty | train | `f980b00bcaf30a971ca0669c8a76372a88dc98281960e20be2f3bc7a7829dbb0` |
| All Beauty | valid | `34a6c71f824eb8ca1ee04250d155458c56b62cbb046b9bd680e09a401999bd20` |
| All Beauty | test | `9d2c8ccdbfc6c4ec0eb59ab73460d094f707f5bd24fe148359e714471049bec8` |
| All Beauty | metadata | `51f8255c2794afd60e274c10d3e2d09dc1f671eba4ba35f74f748d8631216d05` |
| Toys and Games | train | `92e0d9fc488bf663bf0fb0c8c6af21490e3b0c24cc504dea946d9a99972903af` |
| Toys and Games | valid | `2321b056c6571291cf9adaad8ba0709bc00a61becb967f8d232ffbd28a28bf4b` |
| Toys and Games | test | `bc83608596878354b3bcb62b5d246028a60dfd8ff4c99a67fc983557a2dbfa41` |
| Toys and Games | metadata | `c69383112c0c0b0f10439eada942f2cf430511b993d2af8c665a5349c8749f07` |
| Sports and Outdoors | train | `9b553c1c3328dbb7c777bafb9f7e0e2f0b2adf359ef09e516605c719b92ab67a` |
| Sports and Outdoors | valid | `ef0fd9b4f0aa5f3d58e0b9b5aab81b7d570763f476b8ef77be4e7296486776e7` |
| Sports and Outdoors | test | `9b7396cd5c297c00f8abba2a62e3fd1f937ef5255873e16d3eb6df7955fffc1d` |
| Sports and Outdoors | metadata | `f75abcf0af21db0a6c6701f29d1f9b94480557ecbdc238dfc39ebf83a3c9fc3a` |

审计文件：

- `data/amazon2023/processed/All_Beauty/last_out_v2/data_audit.json`
- `data/amazon2023/processed/Toys_and_Games/last_out_v2/data_audit.json`
- `data/amazon2023/processed/Sports_and_Outdoors/last_out_v2/data_audit.json`

## 冻结复现命令

在项目根目录执行：

```bash
# 动态遗忘：三种子、Validation-only、Beam-20/Top-10
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_dynamic_forgetting.py \
  --workers 3 --threads-per-worker 2

# learned hybrid：三种子、Validation 用户哈希 70/30、Test 不参与
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_hybrid_scorer_multiseed.py --workers 3

# 真实外部数据准备、审计、BLaIR 编码和 UniSRec atomic-file 导出
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/prepare_v2_4_external_categories.py --minimum-free-gib 4

# 冻结后一次性 Test 确认，自动生成 Student-t/McNemar/bootstrap
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python scripts/run_v2_4_test_confirmation.py \
  --workers 3 --threads-per-worker 2 --bootstrap-samples 5000
```

唯一配置入口为 `configs/v2_4_adaptive_forgetting.yaml`。动态遗忘 runner 覆盖种子
`2026/2027/2028`；联合 scorer 固定 Beam=20、生成候选=20、BLaIR 检索=50、Top-K=10、
300 个 scorer epoch。UniSRec 曾以 `--workers 1 --threads-per-worker 1 --epochs 10
--train-batch-size 256 --eval-batch-size 128` 启动，但因主动停止，不属于可复现结果命令。

## 正式产物与科学边界

- 动态遗忘：`artifacts/v2_4_adaptive_forgetting/summary.json`
- learned hybrid：`artifacts/v2_4_hybrid_scorer_multiseed/summary.json`
- 外部数据准备：`artifacts/v2_4_external_preparation/state.json`
- UniSRec 停止状态：`artifacts/unisrec_amazon2023_musical_multiseed/state.json`
- 外部验证停止状态：`artifacts/v2_4_external_validation/state.json`
- Test 确认报告：`artifacts/v2_4_test_confirmation/REPORT.md`
- Test 三种子统计：`artifacts/v2_4_test_confirmation/summary.json`
- Test 配对检验：`artifacts/v2_4_test_confirmation/paired_tests.json`

V2.4 的结论是“两个候选增强未通过 Validation 升级门槛，冻结后 Test 也未产生总体或
长尾优势，以及三类真实外部数据已准备完成”。它不提供跨品类模型优势或 UniSRec 对比
结论。最终默认方案仍为原 V2 `token_loss_010 + Beam-20/Top-10`。
