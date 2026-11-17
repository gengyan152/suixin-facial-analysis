# 训练与本机测速

在项目根目录用 PowerShell 执行。所有脚本、缓存、合成样本、权重与日志位于 `training/`；不依赖 `backend/`。SCUT 官方扩展数据已按下列结构就位，90 epoch × 5 折正式训练和真实几何统计已完成，Fold 1 权重已挂载后端。APPA 自训练数据和正式训练不在本次完成范围。

## 环境与固定来源

```powershell
$py = 'python'  # 若使用 Conda / venv，可替换为对应 python.exe 路径
& $py -m pip install --target training/_deps --no-deps --cache-dir training/artifacts/pip_cache -r training/requirements-local.txt
```

基底环境：Python 3.13.7、torch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.3.3。`runtime.py` 自动加入本地依赖路径，并将 torch 下载缓存定位到 `training/artifacts/torch_cache/`。imgaug 保留原官方操作序列，仅补充 NumPy 2 移除的 `sctypes` 类型注册表；输入仍是 BGR，**包括原作者在 BGR 数组上直接调用 imgaug 颜色增强的行为**。运行版本锁见 `artifacts/environment.json`。

源码快照：

- `vendor/2D_FAP`：`9d44888dcad41414a7b7fc55c1d77065f79c1edb`，MIT，保留其 LICENSE。
- `vendor/age-estimation-pytorch`：`4724da9cc87104f64959b9317787860c8e7fc0aa`，MIT，保留其 LICENSE.md。
- `vendor/FPEM`：`c2965425247d7bf8b764d27e4483a06fc7a061e5`，仅用于可选工作量核查；不参与本训练主线。
- 年龄骨干直接载入 `pretrainedmodels==0.7.4` 的原始 `senet.py`，保持官方 `layer0`、`layer1..4`、`last_linear` 键名，不替换成 timm 的近似实现。

年龄 ImageNet 权重原 lip6 主机在本机出现 TLS / HTTP 502，已改用 [MONAI 的同文件镜像](https://github.com/Project-MONAI/MONAI/blob/dev/tests/testing_data/data_config.json)，并核对完整 SHA256 `a260b3a40f82dfe37c58d26a612bcf7bef0d27c6fed096226b0e4e9fb364168e`。这是相同权重的下载入口调整，仍用原 SENet 结构。

## 一键测速与干跑

```powershell
& $py training/benchmark_4060.py --output training/artifacts/benchmark_4060_new.json
& $py training/verify_contracts.py
& $py training/verify_contracts.py --device cuda
& $py training/dry_run.py --workers 2
```

测速为两种完整架构 × FP32/FP16/BF16 × batch 32/64；每项独立子进程，10 步预热，50 步前向、损失、反向和优化器更新，另外 20 步 eval 前向测验证开销。输入已在 GPU 上，**不包含磁盘读图、增强、CPU→GPU 传输和保存权重**。输出包含 images/s、每步时间、`max_memory_allocated`、reserved、空闲显存及 FP16 跳步数；OOM/异常明确保留。关闭游戏、演示服务及其他 GPU 工作后重测，可获得独占 GPU 数值。

干跑使用完整骨干、真实文件读取、Windows spawn DataLoader（2 workers）和合成标签，每个模型训练 2 epoch，导出权重后用新实例 `weights_only=True`、`strict=True` 重新加载，要求完整输出向量 `torch.equal`，误差必须为 0。结果见 `artifacts/dry_run/verification.json`。这是工程烟测，正式精度需在真实数据上评测。

`verify_contracts.py` 运行的是迁移环境中的官方源码对照，不声称已经另外安装并运行 PyTorch 1.8 参考环境。FP32 向量、期望分、分布与损失的逐元素核对结果单独记录。

## SCUT-FBP5500 数据

```text
training/data/SCUT-FBP5500/SCUT-FBP5500_v2/
  aligned/AF1.jpg ...
  train_test_files/5_folders_cross_validations_files/
    cross_validation_1/train_1_aligned_ext.txt
    cross_validation_1/test_1_aligned_ext.txt
    ... cross_validation_5/...
```

每行扩展标注格式严格为：

```text
AF1.jpg 3.45 0.72 2 5 18 25 10
# 文件名 均分 标准差 第1档..第5档评分人数或非负权重；实际文件中不放注释行
```

必须使用含真实均分、标准差和五档分布的官方扩展标注，不从均分编造评分人数。官方代码的数据入口在 `vendor/2D_FAP/data/README.md`，图像/标注需从作者数据包获取。未提供标准差或分布的两列原始划分不能直接用于双标签损失。

```powershell
$scut = 'training/data/SCUT-FBP5500/SCUT-FBP5500_v2'
$splits = "$scut/train_test_files/5_folders_cross_validations_files"
& $py training/check_datasets.py scut --images "$scut/aligned" --splits $splits --output training/artifacts/scut_audit.json
# 完整五折，默认每折 90 epoch
& training/train_five_folds.ps1 -Images "$scut/aligned" -Splits $splits -Precision fp32 -BatchSize 32
# 单折
& $py training/train_2dfap.py --images "$scut/aligned" --train-list "$splits/cross_validation_1/train_1_aligned_ext.txt" --test-list "$splits/cross_validation_1/test_1_aligned_ext.txt" --fold 1 --batch-size 32 --precision fp32
```

模型为 MobileNetV2 + Dropout(0.2) + Linear(1280,40) + Sigmoid + L1 归一化；初始化使用官方 ImageNet V1 权重。细分区间代表值 1.05,1.15,…,4.95；按 `[5,10,10,10,5]` 聚合五档概率。完整分布和期望分写入逐图 CSV。

默认目标分布遵循官方 Laplace CDF 差分，之后 **Sigmoid 再 L1 归一化**；损失为细分分布欧氏距离均值 + 五档分布欧氏距离均值 + `sum(exp(abs(pred-mean))-1)`，默认权重均为 1。可用 `--score-loss l1` 切换期望 MAE 损失，这属于实验变体；`--alpha/--beta/--gamma` 控制三项权重。

官方默认：90 epoch、batch 256、AdamW(lr=.001, weight_decay=.01, amsgrad=True)，每 30 epoch 学习率乘 .1。**本机示例 batch 32 是显存适配实验设置**：batch 改变会改变 BatchNorm、每轮更新次数，以及官方期望损失的 sum 与其他 mean 项的相对作用，不能把小 batch 当成原默认训练完全等价。正式记录配置；严格原 batch 可显式设置 `--batch-size 256`，显存不足则选更大 GPU。没有偷偷使用梯度累积来声称等价。

默认按固定最后一轮导出，无测试集选优。`--dev-fraction 0.1` 只从当前折训练集按固定种子拆分 10% 开发集，按开发 MAE 选权重；实际训练样本数相应减少。测试折仅在训练和选优完成后评测一次。数据切分、配置、选中 epoch 全部记录。

输出 `artifacts/2dfap/foldN/`：`best.pt`、`last.pt`、`config.json`、`split_manifest.json`、`history.json`、`metrics.json`、`predictions.csv`、`roundtrip.json`。五折脚本汇总 PC/MAE/RMSE，报告五折均值、样本标准差（ddof=1）及 pooled 指标，并拒绝测试图跨折重复。

## APPA-REAL 数据与训练

```text
training/data/appa-real-release/
  gt_avg_train.csv          # 4113 行
  gt_avg_valid.csv          # 1500 行
  gt_avg_test.csv           # 1978 行
  train/000000.jpg
  train/000000.jpg_face.jpg
  valid/...                 # 官方 valid 名称
  test/...
```

CSV 使用官方 `file_name,apparent_age_avg,apparent_age_std` 等字段。验证原图与裁剪图存在，沿用原仓库 `ignore_list.csv`，将实际命中的过滤名单写入 `data_audit.json`。缺失或坏图报错，不静默改动划分。

```powershell
$appa = 'training/data/appa-real-release'
& $py training/check_datasets.py appa --data-dir $appa --output training/artifacts/appa_audit.json
& $py training/train_appa_age.py --data-dir $appa --batch-size 32 --precision fp32
# 仅对已有权重做双口径测试
& $py training/train_appa_age.py --data-dir $appa --evaluate training/artifacts/appa/best.pt --output training/artifacts/appa_test
```

官方默认：SE-ResNeXt-50 32×4d、101 类、224×224、80 epoch、batch 128、Adam(lr=.001)、每 20 epoch 学习率乘 .2。当前 CLI 默认 batch 32，其他优化器和日程参数沿用官方；`drop_last=True`，`--keep-last` 才保留末尾不完整 batch。原始样本 4113 不等于过滤后的训练样本量，实际每轮步数以审计结果和 `floor(N/batch)` 为准。

输入：OpenCV 解码 BGR → resize(224,224) → 官方 imgaug 增强 → float32 CHW，**不除以 255，不做 ImageNet 标准化**。训练标签 `clip(round(mean + Normal(0,1)*std*age_stddev),0,100)`，默认倍率 1；验证/测试不加扰动。

按验证集取整标签 MAE 选最佳权重，保持官方选优口径。测试 CSV 包含全部 101 类概率和未取整期望年龄；同时报告取整标签 MAE、原始浮点均值 MAE，以及 `[0,13)、[13,20)、[20,30)、[30,40)、[40,50)、[50,60)、[60,101)` 年龄段两种 MAE、样本数。

官方公开对照权重：<https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth>。2026-11-14 实测 HTTP 200、309024633 bytes。用 `--evaluate` 可加载其中 `state_dict`，无需反序列化整个模型；README 展示过测试 MAE 4.800，这不是本机复现结果。文件名中的 3.9984 也不要当作当前测试 MAE。

## 与后端交接的权重契约

```python
import sys
sys.path.insert(0, 'training')
import torch
from models import TwoDFAP, age_model, score_outputs
model = TwoDFAP()  # 年龄使用 age_model()
checkpoint = torch.load('training/artifacts/2dfap/fold1/best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
```

权重文件只有 `{"state_dict": OrderedDict[str,Tensor]}`，不存模型对象、NumPy 对象或优化器；这是“纯 state_dict 载荷 + 契约要求的 state_dict 外层键”。训练配置、种子、统计单独存 JSON。部署固定第 1 折 `best.pt`；有开发集时它来自折内验证选优，没有开发集时来自第 90 轮。

2D-FAP 输入 RGB、Resize(256)、CenterCrop(224)、ImageNet mean/std；模型输出 `[N,40]`，`score_outputs()` 返回 `[N]` 未取整期望分、`[N,5]` 模型预测评分分布。年龄输入约定如上，输出 logits `[N,101]`，对 dim=1 softmax 后按 0..100 求期望。

> 注：v1.1 已从前后端主链路中移除 478 点像素平面几何与分性别统计模块，下列历史记录保留作为训练实验归档。

## SCUT-FBP5500 正式五折训练（2026-11-14）

作者扩展数据包已下载、校验并解压。ZIP 为 404,749,186 bytes，SHA256：`99be9f950d3a908caf0c4a56da521fcd31680220aa4daa563912e651b8901942`；完整 ZIP 保留在 `training/data/SCUT-FBP5500.zip`。

作者对齐图原名使用 fty/mty/ftw/mtw。根据包内原始划分与 aligned_ext、mtcnn 逐行对应关系，已统一为 AF/AM/CF/CM；仅改图片名与扩展列表首列，所有评分和划分保持不变。映射见 `scut_name_mapping.json`。

数据审计：5500 张（AF 2000、AM 2000、CF 750、CM 750），每折 4400/1100，无坏图、无重叠，每图恰好测试一次。All_Ratings.xlsx 的 330000 条逐人评分与扩展标注五档人数全部精确一致；每图 60 人，标准差采用总体 ddof=0。审计见 `scut_audit.json`、`scut_ratings_audit.json`。

配置：RTX 4060 Laptop 8 GB；PyTorch 2.8.0+cu128；BF16 训练、FP32 最终评测；batch 32、4 workers、seed 20；ImageNet V1 初始化；90 epoch × 5；AdamW lr=0.001、weight_decay=0.01、AMSGrad，每 30 epoch 学习率乘 0.1；三项损失权重均为 1，score_loss=exp。全部使用 4400 张训练图，无折内开发集，固定第 90 轮导出，每折测试集只在训练完成后评测一次。

| Fold | PC | MAE | RMSE | 训练及最终评测耗时 (s) |
|---|---:|---:|---:|---:|
| 1 | 0.914640 | 0.209582 | 0.276547 | 1115.74 |
| 2 | 0.909159 | 0.216889 | 0.289073 | 993.13 |
| 3 | 0.913003 | 0.216457 | 0.284482 | 988.12 |
| 4 | 0.913613 | 0.219415 | 0.284905 | 995.56 |
| 5 | 0.917112 | 0.213639 | 0.274144 | 995.01 |
| 均值 ± 样本标准差 | 0.913505 ± 0.002892 | 0.215196 ± 0.003748 | 0.281830 ± 0.006244 | — |
| Pooled (5500) | 0.912814 | 0.215196 | 0.281886 | — |

五折脚本内训练/保存/最终评测墙钟合计 5087.56 秒（84.79 分钟）；从 Fold 1 配置落盘至 Fold 5 指标落盘跨度 5161.27 秒。下载解压 29.80 秒，几何统计 203.38 秒（与 Fold 1 并行）。从下载开始至后端验收完成 5461.07 秒（91.02 分钟），包括数据核验和各阶段衔接。

各折均含 best.pt、last.pt、config.json、split_manifest.json、history.json、metrics.json、predictions.csv、roundtrip.json；五折 weights_only=True / strict=True 重加载输出误差均为 0。汇总见 `training/artifacts/2dfap/five_fold_summary.json`。

几何统计：训练 female=2174、male=2224；留出 female=574、male=525。CM728、AF1604 姿态超限，AF1995 几何分母退化，共排除 3 张。每项 2000 次配对 bootstrap；训练/留出分别对全部性别、指标及 Pearson/Spearman 做 BH 校正。下表为两种系数均通过 CI/q 检验、且独立留出方向一致的 8 项女性和 9 项男性指标。完整 CI 与 Spearman 见 `gender_stats.json`。

| 分组 | 指标 | 训练 Pearson r | 训练 q(BH) | 留出 Pearson r | 留出 q(BH) |
|---|---|---:|---:|---:|---:|
| female | left_eye_width_face | +0.3297 | 7.29e-56 | +0.3021 | 3.43e-13 |
| female | right_eye_width_face | +0.3671 | 8.76e-70 | +0.3243 | 4.7e-15 |
| female | left_temple_face | -0.1272 | 3.58e-09 | -0.0981 | 0.0242 |
| female | left_brow_eye_ratio | -0.3970 | 3.19e-82 | -0.4291 | 2.98e-26 |
| female | right_brow_eye_ratio | -0.4020 | 2.18e-84 | -0.4471 | 2.17e-28 |
| female | middle_lower_ratio | +0.3307 | 3.54e-56 | +0.2888 | 3.8e-12 |
| female | upper_lower_lip_ratio | -0.3117 | 7.79e-50 | -0.2732 | 5.5e-11 |
| female | mirror_asymmetry | +0.1836 | 9.16e-18 | +0.1377 | 0.00134 |
| male | left_eye_width_face | +0.2898 | 5.6e-44 | +0.3795 | 9.69e-19 |
| male | inner_canthal_face | -0.2257 | 7.3e-27 | -0.2247 | 3.2e-07 |
| male | right_eye_width_face | +0.3221 | 1.97e-54 | +0.3627 | 3.98e-17 |
| male | left_brow_eye_ratio | -0.4187 | 4.51e-94 | -0.3023 | 3.45e-12 |
| male | right_brow_eye_ratio | -0.4434 | 1.38e-106 | -0.3599 | 6.22e-17 |
| male | middle_lower_ratio | +0.2762 | 6.38e-40 | +0.2448 | 2.34e-08 |
| male | nose_inner_canthal_ratio | +0.1082 | 4.06e-07 | +0.1255 | 0.00547 |
| male | upper_lower_lip_ratio | -0.3593 | 3.26e-68 | -0.3983 | 1.14e-20 |
| male | mirror_asymmetry | +0.2612 | 9.37e-36 | +0.2506 | 1.07e-08 |

其中 mirror_asymmetry 越大表示二维镜像偏差越大；本次与评分的正相关按实测保留。

部署：Fold 1 best.pt 与 gender_stats.json 已复制到 backend/weights，并以 SHA256 确认源与目标相同。真实 Pipeline/NumericalModels 验证 beauty_2dfap.weights_status=trained_fold1、weights_ready=true、gender_stats.loaded=true。`pytest -q --live`：43 passed，0 skipped/failed/errors。

集成测试已将旧 weights_ready=false 断言改为正式权重状态，并检查统计加载。首轮套件 42 passed / 1 failed，暴露已有文本解析问题：teeth are not visible 错误覆盖 lips are closed；已将该否定限定于牙齿可见性，随后重跑完整套件。首轮日志及 XML 以 -initial 后缀保留。

可复核汇总、挂载哈希与耗时：`training/artifacts/scut_completion.json`。Android 文件未修改；APPA-REAL 自训练未包含在本次任务中，后端继续使用已有官方年龄权重。
