# 数据与模型准备

仓库默认不提交原始数据、商品向量和训练 checkpoint。这样可以避免把第三方数据或大文件
直接打包进代码仓库，也能保证服务只在模型与数据版本明确时启动。

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[embeddings,serving,benchmark,dev]'
```

## 2. 下载 Amazon Reviews 2023

下载脚本直接访问 McAuley Lab 的 Amazon Reviews 2023 官方公开地址，并生成
`train/valid/test/items/manifest/data_audit` 文件：

```bash
python scripts/prepare_amazon2023.py --download --category Musical_Instruments
python scripts/prepare_amazon2023_v2.py
python scripts/audit_amazon2023.py \
  --processed-dir data/amazon2023/processed/Musical_Instruments/last_out_v2
```

审计必须满足 `status=passed` 且 `synthetic_data_used=false`。正式训练和在线服务不会把
工程 smoke fixture 计入实验结果。

V2 在线输入沿用预处理映射：`0=padding`、`1=1–2 星负向评价`、`2=3 星中性评价`、
`3=4–5 星正向评价`；`behavior_intensity=rating/5`，时间桶由真实时间间隔计算。

## 3. 生成 BLaIR 商品向量

```bash
python scripts/embed_amazon_items.py \
  --backend blair \
  --model hyp1231/blair-roberta-base \
  --input data/amazon2023/processed/Musical_Instruments/last_out/items.jsonl \
  --output data/amazon2023/processed/Musical_Instruments/last_out/item_embeddings_blair.pt
```

向量文件包含 `item_ids`、归一化后的 `embeddings` 和编码元数据。在线长尾模式会检查
`backend=blair`，并要求商品集合与 Semantic-ID catalog 完全一致。

## 4. 训练或放置冻结模型

从头复现默认模型：

```bash
python scripts/train_amazon2023.py --config configs/v2_final_default.yaml
```

在线目录至少需要：

```text
MODEL_DIR/
├── catalog.json
└── generator.pt
```

推荐把完整训练输出放在：

```text
artifacts/amazon2023_musical_instruments_long_tail_training_v1/
└── formal/token_loss_010/seed_2026/
```

若模型来自其他机器，复制后先核对实验报告中的 SHA-256，再运行：

```bash
cp .env.example .env
set -a; source .env; set +a
make check-assets
```

路径检查不通过时服务会停止启动，不会回退到随机权重或虚构商品。

## 5. 容器挂载

镜像不内置数据和 checkpoint；运行时挂载到 `/models` 与 `/data`：

```bash
docker build -t regenrec:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  -v "$PWD/data:/app/data:ro" \
  --env-file .env \
  regenrec:latest
```

浏览 `http://localhost:8000/demo` 查看真实序列演示，或打开 `/docs` 调试接口。
