# D-FINE-S Japan7 RTX 4090 Runbook

## 目标协议

- D-FINE-S + Japan7 + 官方 COCO 权重。
- 单张 RTX 4090，`CUDA_VISIBLE_DEVICES=0`，device `cuda:0`。
- smoke：1 epoch、固定 640×640、train batch 4、val batch 8、workers 4、
  AMP、seed 42。
- 不以 1 epoch AP 高低判定成功。

## 1. 克隆工作分支

```bash
cd /root
git clone \
  --branch codex/japan7-rtx4090-adaptation \
  https://github.com/Leima0214/D-FINE-Japan7.git \
  D-FINE-Japan7
cd /root/D-FINE-Japan7
```

## 2. 安装或检查环境

安装脚本要求 Python 3.11，并创建仓库内 `.venv`。Torch 使用官方
2.5.1、torchvision 0.20.1、CUDA 12.4 wheel；运行脚本会自动使用该
`.venv`。

```bash
bash scripts/setup_rtx4090_env.sh
bash scripts/setup_rtx4090_env.sh --check-only
```

## 3. 检查数据并下载权重

```bash
export JAPAN7_COCO_ROOT=/Japan_COCO/Japan_DFINE_COCO
bash scripts/japan7.sh download
bash scripts/japan7.sh preflight
```

预检必须以 `PREFLIGHT PASS` 结束。它检查 GPU/CUDA、数据、类别协议、
官方权重加载覆盖、模型构建，以及 train/val 各一个 batch，并将 Git
commit、`nvidia-smi` 和 `pip freeze` 写入 `outputs/japan7/preflight/`。

## 4. 运行 1 epoch smoke

```bash
bash scripts/japan7.sh smoke \
  --data-root /Japan_COCO/Japan_DFINE_COCO \
  --device cuda:0 \
  --batch-size 4 \
  --val-batch-size 8 \
  --workers 4 \
  --seed 42
```

脚本保存完整命令、Git commit、依赖、GPU 前后状态、训练日志和 checkpoint。
成功时输出 `SMOKE PASS`；日志包含 NaN、Inf、OOM 或 CUDA error，或未生成
`last.pth` 时返回非零。

若 OOM，只把 `--batch-size` 降为 2；不要删除模型组件或改变算法。

## 5. test-only 与 resume

先定位刚生成的目录：

```bash
SMOKE_DIR="$(find outputs/japan7 -maxdepth 1 -type d \
  -name 'dfine_s_japan7_smoke_*' | sort | tail -n 1)"
test -s "${SMOKE_DIR}/last.pth"
```

独立验证：

```bash
bash scripts/japan7.sh test \
  --checkpoint "${SMOKE_DIR}/last.pth" \
  --data-root /Japan_COCO/Japan_DFINE_COCO \
  --device cuda:0 \
  --val-batch-size 8 \
  --workers 4
```

确认 COCO evaluator 输出 AP50:95、AP50、AP75、APsmall、APmedium 和
APlarge。

状态恢复检查：

```bash
bash scripts/japan7.sh resume \
  --checkpoint "${SMOKE_DIR}/last.pth" \
  --data-root /Japan_COCO/Japan_DFINE_COCO \
  --device cuda:0 \
  --batch-size 4 \
  --val-batch-size 8 \
  --workers 4
```

smoke checkpoint 的 `last_epoch=0`，resume 子命令设置 `epochs=1`，因此
只构建并恢复 model、optimizer、scheduler、warmup、EMA、AMP scaler 和
epoch，不进入下一轮训练。

## 6. 正式训练入口

只有 smoke、test-only 和 resume 都通过后才使用。epoch 必须显式给出；
batch 应根据 smoke 的峰值显存决定。

```bash
bash scripts/japan7.sh formal \
  --epochs <确认后的epoch> \
  --batch-size <确认后的batch> \
  --val-batch-size 8 \
  --workers 4 \
  --seed 42
```

如从 checkpoint 恢复，再加：

```bash
--checkpoint /path/to/last.pth
```

## 7. 打包轻量日志

```bash
tar --exclude='*.pth' -czf japan7_smoke_logs.tar.gz \
  "${SMOKE_DIR}" \
  outputs/japan7/preflight
sha256sum japan7_smoke_logs.tar.gz
```

不要提交或上传数据集、权重、checkpoint、完整 outputs、`.venv` 或密钥。
