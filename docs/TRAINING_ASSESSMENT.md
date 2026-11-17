# RTX 4060 Laptop 训练实测与租卡评估

**SCUT 正式训练已完成**：90 epoch × 5 折，训练及最终评测合计 84.79 分钟；PC=0.913505、MAE=0.215196。完整逐折指标、标准差、pooled 结果与后端验收见文末。下文第 1–9 节保留此前测速、外推和工程验证记录。

更新：2026-11-14。执行设备为本机 NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存；CPU Core Ultra 7 155H，31.5GB RAM。结论：**基础必做部分先在本机训练，不需要为了这一轮课程实验租 RTX 5090。推荐 BF16、batch 32、4 个 DataLoader workers，训练时关闭游戏与演示服务。**

依据这次实测，2D-FAP 五折加 APPA-REAL 年龄训练的 GPU 计算及验证外推合计约 **1.71 小时**；把 JPEG 读取、原增强、CPU→GPU 传输纳入合成图像流水线后，外推约 **2.24 小时**。实际排期建议在这个流水线估计上留 30%—100% 余量，约 **2.9—4.5 小时**，容纳真实图片解码、保存、温度与频率变化。这是一次训练轮数完整、无需连续占用数天的工作量。

该结论采用小 batch 显存适配设置。若验收要求严格保持官方 batch 256/128，而不是保持结构、标签和优化器后记录 batch 变化，则大显存 GPU 有明确用途；尤其 2D-FAP 的 sum 型期望损失使 batch 改变会影响优化行为。

## 1. 实测记录与来源

Material Passport：用户给定第 8 节实施方案；官方源码本地快照；本机实际 PyTorch 执行。完成范围为训练工程、性能测速、合成样本两轮干跑及迁移契约验证。本次已追加完成 SCUT 正式五折训练、精度评测和真实分性别统计，结果见文末；APPA 自训练仍未执行。

| 来源 | 固定版本 / 原始记录 |
|---|---|
| 2D-FAP | `9d44888dcad41414a7b7fc55c1d77065f79c1edb`，`training/vendor/2D_FAP/` |
| 年龄训练 | `4724da9cc87104f64959b9317787860c8e7fc0aa`，`training/vendor/age-estimation-pytorch/` |
| FPEM | `c2965425247d7bf8b764d27e4483a06fc7a061e5`，仅检查可选训练配置 |
| 环境 | `training/artifacts/environment.json` |
| 完整 12 项矩阵 | `training/artifacts/benchmark_final.json`；2D-FAP 原始项为 `benchmark_2dfap_corrected_*`，年龄为 `benchmark_final_appa_*` |
| 推荐配置复测 | `training/artifacts/benchmark_repeat.json` |
| 数据流水线 | `training/artifacts/pipeline_2dfap.json`、`pipeline_appa.json` |
| 工程验证 | `training/artifacts/contracts_verification.json`、`contracts_verification_cuda.json`、`dry_run/verification.json` |

环境：`conda` `torch_env`（Python 3.13.7、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、CUDA runtime 12.8、驱动 595.79、Windows WDDM）。硬件快照见 `training/artifacts/hardware_snapshot.txt`。笔记本功耗限额会动态变化，过程中观察过约 46—50W 的 cap，采样功耗也有超过 50W 的时刻；本报告不把它当作满功耗台式 4060。

最初检测到 YuanShen / ZFGameBrowser 占用 GPU，早期 FP32 / AMP 出现明显波动和较大 batch 停滞，已停止该轮并保留 `benchmark_4060*`、`benchmark_stable*` 原始记录，不纳入推荐时间计算。后来游戏退出，完成下表。并行开发期间也出现过其他 Python GPU 上下文，所以对推荐配置另外复测，并保存起止 GPU PID 清单；原始矩阵用于比较实测条件，推荐时间以复测和流水线为依据。

## 2. 官方训练默认值核实

| 项目 | 2D-FAP | APPA-REAL 年龄 |
|---|---|---|
| 骨干 / 参数量 | MobileNetV2 / 2,275,112 | 原 Cadene SE-ResNeXt-50 32×4d / 25,717,845 |
| 输出 | 40 维 Sigmoid + L1 分布，五档聚合 | 101 类 logits，softmax 后求 0—100 期望 |
| 输入 | RGB，Resize 256 + CenterCrop 224，ImageNet mean/std | OpenCV BGR，resize 224×224，float32，0—255 |
| 官方 epoch / batch | **90 / 256** | **80 / 128** |
| 优化器 | AdamW，lr .001，betas (.9,.999)，eps 1e-8，AMSGrad，默认 weight_decay .01 | Adam，lr .001，默认 betas/eps，weight_decay 0 |
| LR 日程 | 每 30 epoch × .1 | 每 20 epoch × .2 |
| 数据 | 5500；每折 4400 训练 / 1100 测试，5 折 | 原始 4113 训练 / 1500 验证 / 1978 测试 |
| 标签 | 均值、std 的 Laplace CDF 差分分布 + 真实五档分布 + 期望损失 | round(mean + Normal(0,1) × std)，截到 0—100 |
| 训练尾批 | 保留尾批 | 官方 `drop_last=True` |
| 选优 | 官方固定最后一轮；实现支持训练折内部开发集 | 验证集取整年龄 MAE 最佳 |

出处：[2D-FAP 训练脚本](https://github.com/enquan/2D_FAP/blob/9d44888dcad41414a7b7fc55c1d77065f79c1edb/2D_FAP.py)、[模型](https://github.com/enquan/2D_FAP/blob/9d44888dcad41414a7b7fc55c1d77065f79c1edb/models/MobileNetv2.py)、[年龄 defaults](https://github.com/yu4u/age-estimation-pytorch/blob/4724da9cc87104f64959b9317787860c8e7fc0aa/defaults.py)、[年龄 dataset](https://github.com/yu4u/age-estimation-pytorch/blob/4724da9cc87104f64959b9317787860c8e7fc0aa/dataset.py)、[APPA 数据说明](https://chalearnlap.cvc.uab.es/dataset/26/description/)。

2D-FAP 不使用 Softmax 替换分布头。40 个分数中心为 1.05—4.95，五档聚合区间计数 `[5,10,10,10,5]`。三项损失分别为细分分布欧氏距离均值、五档分布欧氏距离均值、`sum(exp(abs(score-mean))-1)`，权重均为 1。小 batch 会改变第三项相对于前两项的权重尺度、BN 统计和每轮更新次数，因此正式训练配置与指标一起保存。AMP 也属于数值精度变体，应在真实验证集核验。

年龄脚本严格保持 BGR 0—255，未除以 255；保留官方 imgaug 操作序列。沿用 `ignore_list.csv`，实际过滤后的样本数由 `check_datasets.py` / `data_audit.json` 输出。下文按 4113/1500/1978 原始划分估算，过滤后的真实步数通常更少，不能在未取得 CSV 时编造过滤后人数。

## 3. 完整 GPU 训练矩阵

每项都构造完整模型、全部可训练参数和官方优化器；使用 GPU 上的合成输入与标签，10 步预热后测 50 次前向+损失+反向+优化器更新。FP16 使用 GradScaler，接受结果要求实测段没有跳过更新；BF16 不需要 GradScaler。TF32 关闭、cuDNN 自动算法搜索关闭，NCHW；不使用编译、channels_last 或梯度检查点。每项独立子进程，释放上一项 CUDA 状态。

wall 时间按 CUDA 同步后的 `perf_counter` 计算，原始每步 CUDA Event 时间也保存；峰值为预热后 reset 的 `torch.cuda.max_memory_allocated()`。reserved、驱动占用、其他进程显存另列，**allocated 并不等于整张显卡的总占用**。GB 表格使用 GiB（1024³ bytes）。

| 模型 | 精度 | Batch | images/s | wall ms/step | peak allocated GiB | peak reserved GiB | eval images/s | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2dfap | fp32 | 32 | 306.75 | 104.32 | 2.471 | 2.744 | 1213.63 | 50/50 更新 |
| 2dfap | fp32 | 64 | 310.97 | 205.81 | 4.806 | 5.369 | 1181.02 | 50/50 更新 |
| 2dfap | fp16 | 32 | 624.85 | 51.21 | 1.331 | 1.445 | 2653.32 | 50/50 更新 |
| 2dfap | fp16 | 64 | 625.17 | 102.37 | 2.512 | 2.781 | 2456.33 | 50/50 更新 |
| 2dfap | bf16 | 32 | 476.15 | 67.21 | 1.331 | 1.445 | 2134.99 | 50/50 更新 |
| 2dfap | bf16 | 64 | 484.57 | 132.08 | 2.512 | 2.781 | 2116.92 | 50/50 更新 |
| appa | fp32 | 32 | 56.92 | 562.16 | 4.417 | 4.871 | 260.48 | 50/50 更新 |
| appa | fp32 | 64 | — | — | — | — | — | timeout |
| appa | fp16 | 32 | 192.24 | 166.46 | 2.483 | 2.818 | 498.18 | 50/50 更新 |
| appa | fp16 | 64 | 141.23 | 453.17 | 4.538 | 4.799 | 442.17 | 50/50 更新 |
| appa | bf16 | 32 | 182.84 | 175.02 | 2.483 | 2.818 | 680.51 | 50/50 更新 |
| appa | bf16 | 64 | 192.73 | 332.07 | 4.538 | 4.945 | 648.66 | 50/50 更新 |

OOM 表示分配失败；timeout 表示在 300 秒的单项上限内未完成，可能涉及 Windows 显存回退/换页，不换算为成功吞吐量。已完成项的 50 个训练更新均通过非有限 loss 与 FP16 跳步检查。

推荐配置另行复测：

| 模型 | 精度 / Batch | images/s | peak allocated GiB | eval images/s |
|---|---|---:|---:|---:|
| 2dfap | BF16 / 32 | 476.15 | 1.331 | 2134.99 |
| appa | BF16 / 32 | 182.06 | 2.483 | 670.02 |

## 4. JPEG、增强和传输实测

`benchmark_pipeline.py` 使用 256 张 256×280 合成 JPEG、真实 Dataset、原增强序列、Windows 4 workers、pin_memory、CPU→GPU 传输、完整训练更新；10 步预热、50 步实测。固定种子并开启确定性算法，设置与训练脚本一致。第二轮读取会受文件系统缓存帮助，这也符合多 epoch 反复读取小型数据集的情形。真实人脸的压缩率、分辨率与增强耗时仍可能不同。

| 模型 | 精度 / Batch | workers | 包含数据处理的 images/s | peak allocated GiB |
|---|---|---:|---:|---:|
| 2dfap | bf16 / 32 | 4 | 346.43 | 1.312 |
| appa | bf16 / 32 | 4 | 153.97 | 2.476 |

## 5. 完整训练耗时计算

设训练吞吐量为 R，验证吞吐量为 V，平均训练步时间 t=B/R。样本线性估计为 N×E/R；按代码步数估计：2D-FAP `ceil(N/B)×E×t`，年龄 `floor(N_filtered/B)×E×t`（drop_last）。尾批时间不一定与满 batch 完全相同，2D-FAP 满 batch 步数估计略偏保守。以下全部为 90/80 个完整 epoch，未用早停缩短时间。

在 B=32 下，2D-FAP 每折 138×90 = **12420 步**；APPA 原始 4113 张、忽略名单尚未扣除时，每轮 128 步、每轮实际使用 4096 张，80 轮为 **10240 步**。使用原始浮点样本量线性估计，2D-FAP 单折为 **13.86 分钟**，年龄为 **30.12 分钟**。

| 任务 | 推荐配置 GPU 步数外推 | 加入数据流水线的外推 | 所用峰值显存 |
|---|---:|---:|---:|
| 2D-FAP 单折训练 90 epoch | 13.91 分钟 | 19.12 分钟 | 1.331 GiB |
| 2D-FAP 单折最终 1100 测试前向 | 0.01 分钟 | 按相同 GPU 前向时间计；未单测测试读图 | — |
| 2D-FAP 完整 5 折训练 + 每折最终测试 | 69.60 分钟 | 95.65 分钟 | 同单折，五折串行不叠加 |
| APPA 80 epoch 训练 | 30.00 分钟 | 35.47 分钟 | 2.483 GiB |
| APPA 每轮 1500 验证 + 一次 1978 测试前向 | 3.03 分钟（BF16 测速口径） / 约 8.04 分钟（正式脚本 FP32 验证口径） | 按 FP32 评测口径外推合计约 **2.32 小时**（BF16 口径 2.24 小时） | — |
| 基础全部 | **1.71 小时（BF16） / 1.79 小时（FP32 验证）** | **2.24—2.32 小时** | 两模型串行训练 |

完整墙钟时长还包括验证读图、启动 workers、保存最佳权重、日志及温度变化，所以排期采用 2.9—4.5 小时，而不把上表秒数当成整轮真实训练已经跑完。若启用 10% 折内开发集，则训练变成 3960 张、每轮多一次 440 张开发评测，需按实际步数和日志重新计算。

## 6. 公开年龄权重

官方 demo 指向 [v1.0 Release 权重](https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth)，本机 HEAD 实测返回 **200**、Content-Length **309024633 bytes**，证据在 `training/artifacts/age_release_probe.json`。因此可以直接作为公开 APPA 模型对照或先做接口闭环；仍按方案完成自己的年龄训练。

加载使用 `torch.load(..., weights_only=True)` 后 `checkpoint["state_dict"]`，严格校键。官方 README 示例测试 MAE 为 4.800；这属于作者示例，文件名 3.9984 也不替代本地双口径测试。新脚本输出取整标签 MAE、原始 apparent_age_avg MAE 和年龄段误差。

## 7. 是否租 RTX 5090

**这一轮基础交付不必租。** 两个模型在小 batch AMP 下均完成真实训练迭代，显存余量明显；五折与年龄合计是几小时量级。优先用现成环境把真实数据、切分和指标跑完整，收益大于再搭一套云端环境。

云端总时长应算 `上传/下载数据 + 配环境 + 本地训练时间/S + 下载权重`，S 是同一脚本在租用实例上的实测加速比。本任务没有租 5090，因此不虚构 S；[桌面 RTX 5090 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) 为 32GB GDDR7，具体云商品仍须确认型号与可用显存。

上传示例（场景假设，不是本机实测网速）：5GiB 在有效 20Mbps 下约 35.8 分钟，100Mbps 下约 7.2 分钟；10GiB 分别约 71.6 / 14.3 分钟。解压、依赖、权重下载和首个 batch 排错另预留 20—60 分钟。已有云镜像与云盘数据时这部分可以大幅缩短。

若假设 S=3，纯计算节省是本地时长的 2/3；只有迁移开销低于 `T_local×(1−1/S)` 才缩短总等待。以数据流水线外推 2.24 小时计，这个阈值约 **89 分钟**。因此已有环境和数据的云卡可能省时，但不是基础任务能否完成的必要条件。

值得租用的具体情况：

1. 要严格运行官方 batch 256/128，或扩展输入分辨率且实际出现 OOM；更大显存直接解决容量问题。
2. 要反复跑多个随机种子、多个损失组合、FPEM 三阶段对照，形成几十小时本地任务，而截止时间紧迫。
3. FaceLLM 微调需要多图块、长序列或解冻视觉骨干，batch 1 + 检查点仍无法在 8GB 完成真实训练步。
4. 笔记本必须同时用于演示/游戏，不能留出几小时独占 GPU；此时租卡换来的是资源隔离。

## 8. 可选 FPEM 与 FaceLLM-1B

FPEM 官方当前公开配置并非一次简单 MobileNet 训练：`public_swin_ca` 为 **50 epoch / batch 32**，冻结人脸先验分支、训练 Swin 骨干；`public_faclip` 为 **10 epoch / batch 16**；`public_fpem` 融合阶段为 **50 epoch / batch 16**，配置中冻结两个分支。配置使用 LiveBeauty 路径，迁到 SCUT 五折还需保持同一切分并重新训练分支。出处：[FPEM 官方 experiments](https://github.com/Estella-LH/FPEM/tree/c2965425247d7bf8b764d27e4483a06fc7a061e5/experiments)。

以 SCUT 每折 4400 张、三个阶段都从该折训练算，五折合计 `4400×5×(50+10+50)=2,420,000` 次训练样本暴露，且各阶段单图开销不同。其时间公式是 `22000×(50/R_swin+10/R_clip+50/R_fusion)` 秒，不能直接套用 2D-FAP 吞吐量。本任务没有运行 FPEM 三阶段，所以未给出伪实测显存/小时数。8GB 可先尝试小 batch AMP、冻结先验分支和融合模块训练；正式做完整多种子五折对照时，5090 的显存与周转速度更有价值。

[FaceLLM-1B 模型卡](https://huggingface.co/Idiap/FaceLLM-1B) 给出约 0.9B 参数、InternVL3-1B 骨干，实际目录是 `model/`。BF16 冻结权重的静态预算约 1.8GB（十进制）；LoRA 只为少量适配器保留梯度和优化器状态，但图像 token、文本长度、视觉模块是否冻结会决定激活峰值。8GB 可尝试 **batch 1、单图块、短序列、梯度检查点、冻结视觉模块** 的小型 LoRA；是否装得下必须通过目标配置的真实反向步确认。

全参数 Adam 混合精度训练按常见权重/梯度/主权重/两份动量约 16 bytes/参数估计，0.9B 已约 14.4GB，尚未包含激活，因此不适合这张 8GB 卡。LoRA 的时间必须用真实图文样本实测 `N×epochs/有效样本吞吐量`；例如 1000 样本×3 epoch 在假设 0.5 / 2 samples/s 时分别为 100 / 25 分钟，**这两个速率只是预算场景，不是本机 FaceLLM 测速**。当前方案首版明确不微调 FaceLLM，所以它不构成本轮租卡理由。

## 9. 工程验收与使用入口

| 检查 | 本次结果 |
|---|---|
| 官方 MobileNet 源码 vs 迁移模型 | CPU 与 CUDA 的 FP32 完整向量、期望、五档分布、loss 精确一致 |
| 官方年龄 factory vs 迁移模型 | CPU 与 CUDA 的 logits / softmax 完整向量精确一致 |
| BGR 0—255 / 原浮点标签 / 取整标签 | 哨兵像素和标签检查通过 |
| 非正方形像素几何与 Roll 不变性 | 最大差约 2.44e-15 |
| 2D-FAP 2 epoch → 保存 → 新实例加载 | exact_equal=True，最大差 0.0 |
| 年龄 2 epoch → 保存 → 新实例加载 | exact_equal=True，最大差 0.0 |
| Windows DataLoader 多进程 | 两模型均 2 workers 干跑通过 |
| 分性别统计 | 合成 8 训练 / 4 留出贯通，未把不足样本的趋势标为可展示 |
| 五折汇总算术 | 合成预测的均值、样本标准差、pooled RMSE 与解析公式一致 |
| 原生 MediaPipe 提取入口 | 固定 v1 task、CPU 执行，12 张随机噪声全部记录为无脸 |
| ImageNet 初始化 | 两骨干下载并 strict 加载成功；SENet 镜像完整 SHA256 已核对 |

迁移对照使用现代 PyTorch 中运行的官方源码，只改废弃 import；没有另建 PyTorch 1.8 环境。权重全部为 tensor-only `{"state_dict": ...}`，未序列化模型对象，加载 `weights_only=True` 与 `strict=True`。

交付脚本、数据结构、单折/五折训练、数据核查、几何 JSON 字段、索引表与后端权重加载方式详见 [training/README.md](../training/README.md)。SCUT 已按数据审计、训练、统计、挂载顺序完成；下节记录本次正式实测结果。

## SCUT-FBP5500 正式五折训练与挂载（2026-11-14）

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

部署：Fold 1 best.pt 与 gender_stats.json 已复制到 backend/weights，并以 SHA256 确认源与目标相同。真实 Pipeline/NumericalModels 验证 beauty_2dfap.weights_status=trained_fold1、weights_ready=true、gender_stats.loaded=true。`pytest -q --live`：43 passed，0 skipped/failed/errors；报告为 `backend/scut-trained-test-results.xml`、`backend/live_validation.json`。

集成测试已将旧 weights_ready=false 断言改为正式权重状态，并检查统计加载。首轮套件 42 passed / 1 failed，暴露已有文本解析问题：teeth are not visible 错误覆盖 lips are closed；已将该否定限定于牙齿可见性，随后重跑完整套件。首轮日志及 XML 以 -initial 后缀保留。

可复核汇总、挂载哈希与耗时：`training/artifacts/scut_completion.json`。Android 文件未修改；APPA-REAL 自训练未包含在本次任务中，后端继续使用已有官方年龄权重。
