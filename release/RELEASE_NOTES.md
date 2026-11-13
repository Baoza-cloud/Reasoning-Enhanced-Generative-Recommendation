# ReGenRec V2.4 frozen model

- 数据：Amazon Reviews 2023 / Musical Instruments；
- 默认方案：`token_loss_010 + Beam-20/Top-10`；
- 随机种子：2026；
- 商品数：24,587；
- Semantic ID：3 层 RQ code + 1 个碰撞消歧 token；
- 解码：Catalog Trie constrained Beam Search；
- 发布包不包含 Amazon 原始数据或 BLaIR 商品向量。

完整实验边界、三种子结果与负结果见 `docs/V2_FINAL_REPORT.md`。
