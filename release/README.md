# 冻结模型发布

## 生成发布包

```bash
make package-model
```

命令生成：

```text
release_assets/
├── regenrec-v2.4-seed2026-model.tar.gz
└── regenrec-v2.4-seed2026-model.tar.gz.sha256
```

归档只包含 `generator.pt`、`catalog.json`、冻结配置和模型清单。Amazon 数据与 BLaIR 商品
向量不进入发布包，按 `docs/DATA_AND_MODEL_SETUP.md` 单独准备。

## 发布到 GitHub Release

安装并登录 GitHub CLI 后执行：

```bash
gh release create v2.4 \
  release_assets/regenrec-v2.4-seed2026-model.tar.gz \
  release_assets/regenrec-v2.4-seed2026-model.tar.gz.sha256 \
  --title "ReGenRec V2.4 frozen model" \
  --notes-file release/RELEASE_NOTES.md
```

发布后把 Release 下载地址填入项目 README。不要把 checkpoint 直接提交到 Git 历史。

## 下载与校验

```bash
curl -L -o model.tar.gz RELEASE_ASSET_URL
curl -L -o model.tar.gz.sha256 RELEASE_CHECKSUM_URL
shasum -a 256 -c model.tar.gz.sha256
tar -xzf model.tar.gz
```
