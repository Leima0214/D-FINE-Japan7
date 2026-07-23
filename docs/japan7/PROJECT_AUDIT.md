# D-FINE-S Japan7 工程审计

## 结论

本适配保留官方 D-FINE-S 模型、损失、增强、优化器和学习率，只覆盖 Japan7
数据路径、`num_classes=7`、类别协议和单卡运行参数。配置继承
`configs/dfine/dfine_hgnetv2_s_coco.yml`，没有重写模型。

本地只完成静态、CPU 构建、数据批次和预训练权重兼容性检查；RTX 4090、
CUDA AMP、完整 1 epoch、test-only 和 resume 仍须在租赁 GPU 上实测。

## 审计表

| 检查项 | 证据/当前行为 | 风险与处理 | 本地结果 |
| --- | --- | --- | --- |
| D-FINE-S 配置 | `dfine_hgnetv2_s_coco.yml` 依次包含 COCO 数据、runtime、dataloader、optimizer 和完整 D-FINE/HGNetv2 定义 | Japan7 配置只继承并覆盖必要字段 | PASS |
| 配置优先级 | `yaml_utils.load_config()` 先按 `__include__` 顺序合并，子配置最后覆盖；CLI `-u` 最后覆盖 YAML | 复用现有 CLI，不新增环境变量解析器 | PASS |
| 类别数 | `num_classes` 是模型、criterion 和 postprocessor 的共享配置 | Japan7 固定为 7 | PASS |
| category id | `CocoDetection` 在 `remap_mscoco_category=False` 时直接使用 JSON 的 id；postprocessor 同样不映射 | 数据必须严格使用连续 0–6 | PASS |
| 数据路径 | train/val 的 `img_folder` 与 `ann_file` 位于 dataloader dataset 配置 | 仓库只保存远程默认路径；本地/其他挂载由 `-u` 覆盖 | PASS |
| COCO 微调 | 参数为 `-t/--tuning`；仅加载同名同形状 tensor | 官方 80 类分类头与 Japan7 7 类头不匹配时重新初始化 | PASS |
| 权重覆盖 | 官方 `dfine_s_coco.pth` 顶层为 `model` | 按参数量覆盖 99.91%；仅分类/去噪头未加载 | PASS |
| resume | 参数为 `-r/--resume`；训练路径构建 optimizer、scheduler、warmup、EMA 和 scaler 后调用 `load_state_dict`，同时恢复 `last_epoch` | resume 子命令将 `epochs=1`，对 smoke 的 epoch 0 checkpoint 只加载状态而不继续训练 | 待 GPU 生成 checkpoint |
| test-only | `--test-only -r checkpoint` 进入 `solver.val()` | 输出由 COCO evaluator 给出 AP50:95、AP50、AP75、APsmall、APmedium、APlarge | 待 GPU checkpoint |
| 单 GPU | 无 `RANK/LOCAL_RANK/WORLD_SIZE` 时分布式初始化失败后回退，world size 为 1 | 可直接 `python train.py`，不需要 `torchrun` | PASS（CPU 路径） |
| batch 语义 | `total_batch_size / world_size` 得到每 rank batch | 单卡时全局 batch 等于单卡 batch | PASS |
| AMP | scaler 存在时使用 `torch.autocast` | 原代码把 `cuda:0` 误作 device type；已统一改为 `device.type` | PASS（静态/CPU） |
| SyncBatchNorm | 仅分布式初始化后才转换 | Japan7 配置显式关闭 | PASS |
| torch.compile | `warp_model(..., compile=False)`，配置未开启 | RTX 4090 smoke 不启用 | PASS |
| FlashAttention | 模型与依赖均未要求 | 不安装、不启用 | PASS |
| Python/Torch | 官方工程建议 Python 3.11；PyTorch 官方提供 2.5.1/torchvision 0.20.1 的 cu124 wheel | 环境脚本只固定这对运行时，其余遵循 `requirements.txt` | 待远程安装 |

## 本地数据与模型证据

- train：9,451 images，22,241 annotations。
- val：1,051 images，2,507 annotations。
- 类别：`0:D00, 1:D10, 2:D20, 3:D40, 4:D43, 5:D44, 6:D50`。
- JSON、image/annotation 引用、bbox 格式与正宽高、图片存在性检查通过。
- D-FINE-S CPU 构建成功：10,228,145 parameters。
- train/val dataset 构建成功，并各读取一个 batch；标签范围为 0–6。
- 官方权重：41,841,422 bytes，
  SHA256 `48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89`。

## 采纳与舍弃

- 采纳：只读 COCO 检查、官方配置继承、官方权重审计、单卡/AMP 根因修复、
  环境安装、远程预检、smoke、test-only、resume、日志和复现信息保存。
- 舍弃单独的 resolved YAML 工具：项目已有 `-u` 嵌套配置覆盖。
- 舍弃 smoke/formal 两套重复 YAML：同一 Japan7 配置由运行脚本覆盖 batch、epoch 和路径。
- 合并多个薄脚本为 `scripts/japan7.sh` 子命令；合并数据、smoke、复现文档到一个 Runbook。
- 不安装 `pycocotools`：当前工程实际使用 `faster-coco-eval`。
- 不预设正式训练 epoch；RTX 4090 smoke 完成后再决定。

## 本机限制

Windows base Python 3.12.7 在导入 Torch 时存在重复 Intel OpenMP runtime
错误。本地验证改用已有的可用环境完成；没有使用
`KMP_DUPLICATE_LIB_OK` 绕过。该问题不影响独立远程 `.venv`，也不计作
RTX 4090 验证。
