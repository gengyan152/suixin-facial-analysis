"""Generate the assessment from recorded measurements, never from GPU name guesses."""
import json
import math
from pathlib import Path
from runtime import ROOT


def main():
    artifacts = ROOT/'artifacts'
    rows = json.loads((artifacts/'benchmark_final.json').read_text(encoding='utf-8'))
    repeats = json.loads((artifacts/'benchmark_repeat.json').read_text(encoding='utf-8'))
    chosen = {name: next(r for r in repeats if r['model'] == name and r['status'] == 'ok') for name in ['2dfap', 'appa']}
    pipelines = {name: json.loads((artifacts/f'pipeline_{name}.json').read_text(encoding='utf-8')) for name in ['2dfap', 'appa']}
    verification = json.loads((artifacts/'dry_run/verification.json').read_text(encoding='utf-8'))
    table = []
    for r in sorted(rows, key=lambda r: (r['model'], ['fp32','fp16','bf16'].index(r['precision']), r['batch'])):
        if r['status'] == 'ok':
            table.append(f"| {r['model']} | {r['precision']} | {r['batch']} | {r['images_per_second']:.2f} | {r['seconds']/r['steps']*1000:.2f} | {r['peak_allocated_mib']/1024:.3f} | {r['peak_reserved_mib']/1024:.3f} | {r['eval_images_per_second']:.2f} | 50/50 更新 |")
        else:
            table.append(f"| {r['model']} | {r['precision']} | {r['batch']} | — | — | — | — | — | {r['status']} |")
    repeat_table = '\n'.join(f"| {name} | BF16 / 32 | {r['images_per_second']:.2f} | {r['peak_allocated_mib']/1024:.3f} | {r['eval_images_per_second']:.2f} |" for name, r in chosen.items())
    pipeline_table = '\n'.join(f"| {name} | {r['precision']} / {r['batch']} | {r['workers']} | {r['images_per_second']:.2f} | {r['peak_allocated_mib']/1024:.3f} |" for name, r in pipelines.items())
    f, a = chosen['2dfap'], chosen['appa']
    pf, pa = pipelines['2dfap'], pipelines['appa']
    f_steps = math.ceil(4400/32)*90
    a_steps = math.floor(4113/32)*80
    f_time = f_steps*32/f['images_per_second']/60
    f_test = 1100/f['eval_images_per_second']/60
    a_time = a_steps*32/a['images_per_second']/60
    a_eval = (1500*80+1978)/a['eval_images_per_second']/60
    pf_time = f_steps*32/pf['images_per_second']/60
    pa_time = a_steps*32/pa['images_per_second']/60
    compute_hours = (5*(f_time+f_test)+a_time+a_eval)/60
    pipeline_hours = (5*(pf_time+f_test)+pa_time+a_eval)/60
    text = f'''# RTX 4060 Laptop 训练实测与租卡评估

更新：2026-11-14。执行设备为本机 NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存；CPU Core Ultra 7 155H，31.5GB RAM。结论：**基础必做部分先在本机训练，不需要为了这一轮课程实验租 RTX 5090。推荐 BF16、batch 32、4 个 DataLoader workers，训练时关闭游戏与演示服务。**

依据这次实测，2D-FAP 五折加 APPA-REAL 年龄训练的 GPU 计算及验证外推合计约 **{compute_hours:.2f} 小时**；把 JPEG 读取、原增强、CPU→GPU 传输纳入合成图像流水线后，外推约 **{pipeline_hours:.2f} 小时**。实际排期建议在这个流水线估计上留 30%—100% 余量，约 **{pipeline_hours*1.3:.1f}—{pipeline_hours*2:.1f} 小时**，容纳真实图片解码、保存、温度与频率变化。这是一次训练轮数完整、无需连续占用数天的工作量。

该结论采用小 batch 显存适配设置。若验收要求严格保持官方 batch 256/128，而不是保持结构、标签和优化器后记录 batch 变化，则大显存 GPU 有明确用途；尤其 2D-FAP 的 sum 型期望损失使 batch 改变会影响优化行为。

## 1. 实测记录与来源

Material Passport：用户给定第 8 节实施方案；官方源码本地快照；本机实际 PyTorch 执行。完成范围为训练工程、性能测速、合成样本两轮干跑及迁移契约验证。真实数据未放入本交付目录，因此尚未进行 SCUT/APPA 全量训练、正式精度测评或真实分性别相关性统计。

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
{chr(10).join(table)}

OOM 表示分配失败；timeout 表示在 300 秒的单项上限内未完成，可能涉及 Windows 显存回退/换页，不换算为成功吞吐量。已完成项的 50 个训练更新均通过非有限 loss 与 FP16 跳步检查。

推荐配置另行复测：

| 模型 | 精度 / Batch | images/s | peak allocated GiB | eval images/s |
|---|---|---:|---:|---:|
{repeat_table}

## 4. JPEG、增强和传输实测

`benchmark_pipeline.py` 使用 256 张 256×280 合成 JPEG、真实 Dataset、原增强序列、Windows 4 workers、pin_memory、CPU→GPU 传输、完整训练更新；10 步预热、50 步实测。固定种子并开启确定性算法，设置与训练脚本一致。第二轮读取会受文件系统缓存帮助，这也符合多 epoch 反复读取小型数据集的情形。真实人脸的压缩率、分辨率与增强耗时仍可能不同。

| 模型 | 精度 / Batch | workers | 包含数据处理的 images/s | peak allocated GiB |
|---|---|---:|---:|---:|
{pipeline_table}

## 5. 完整训练耗时计算

设训练吞吐量为 R，验证吞吐量为 V，平均训练步时间 t=B/R。样本线性估计为 N×E/R；按代码步数估计：2D-FAP `ceil(N/B)×E×t`，年龄 `floor(N_filtered/B)×E×t`（drop_last）。尾批时间不一定与满 batch 完全相同，2D-FAP 满 batch 步数估计略偏保守。以下全部为 90/80 个完整 epoch，未用早停缩短时间。

在 B=32 下，2D-FAP 每折 138×90 = **{f_steps} 步**；APPA 原始 4113 张、忽略名单尚未扣除时，每轮 128 步、每轮实际使用 4096 张，80 轮为 **{a_steps} 步**。使用原始浮点样本量线性估计，2D-FAP 单折为 **{4400*90/f['images_per_second']/60:.2f} 分钟**，年龄为 **{4113*80/a['images_per_second']/60:.2f} 分钟**。

| 任务 | 推荐配置 GPU 步数外推 | 加入数据流水线的外推 | 所用峰值显存 |
|---|---:|---:|---:|
| 2D-FAP 单折训练 90 epoch | {f_time:.2f} 分钟 | {pf_time:.2f} 分钟 | {max(f['peak_allocated_mib'],pf['peak_allocated_mib'])/1024:.3f} GiB |
| 2D-FAP 单折最终 1100 测试前向 | {f_test:.2f} 分钟 | 按相同 GPU 前向时间计；未单测测试读图 | — |
| 2D-FAP 完整 5 折训练 + 每折最终测试 | {5*(f_time+f_test):.2f} 分钟 | {5*(pf_time+f_test):.2f} 分钟 | 同单折，五折串行不叠加 |
| APPA 80 epoch 训练 | {a_time:.2f} 分钟 | {pa_time:.2f} 分钟 | {max(a['peak_allocated_mib'],pa['peak_allocated_mib'])/1024:.3f} GiB |
| APPA 每轮 1500 验证 + 一次 1978 测试前向 | {a_eval:.2f} 分钟 | 按相同 GPU 前向时间计；未单测评测读图 | — |
| 基础全部 | **{compute_hours:.2f} 小时** | **{pipeline_hours:.2f} 小时** | 两模型串行训练 |

完整墙钟时长还包括验证读图、启动 workers、保存最佳权重、日志及温度变化，所以排期采用 {pipeline_hours*1.3:.1f}—{pipeline_hours*2:.1f} 小时，而不把上表秒数当成整轮真实训练已经跑完。若启用 10% 折内开发集，则训练变成 3960 张、每轮多一次 440 张开发评测，需按实际步数和日志重新计算。

## 6. 公开年龄权重

官方 demo 指向 [v1.0 Release 权重](https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth)，本机 HEAD 实测返回 **200**、Content-Length **309024633 bytes**，证据在 `training/artifacts/age_release_probe.json`。因此可以直接作为公开 APPA 模型对照或先做接口闭环；仍按方案完成自己的年龄训练。

加载使用 `torch.load(..., weights_only=True)` 后 `checkpoint["state_dict"]`，严格校键。官方 README 示例测试 MAE 为 4.800；这属于作者示例，文件名 3.9984 也不替代本地双口径测试。新脚本输出取整标签 MAE、原始 apparent_age_avg MAE 和年龄段误差。

## 7. 是否租 RTX 5090

**这一轮基础交付不必租。** 两个模型在小 batch AMP 下均完成真实训练迭代，显存余量明显；五折与年龄合计是几小时量级。优先用现成环境把真实数据、切分和指标跑完整，收益大于再搭一套云端环境。

云端总时长应算 `上传/下载数据 + 配环境 + 本地训练时间/S + 下载权重`，S 是同一脚本在租用实例上的实测加速比。本任务没有租 5090，因此不虚构 S；[桌面 RTX 5090 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) 为 32GB GDDR7，具体云商品仍须确认型号与可用显存。

上传示例（场景假设，不是本机实测网速）：5GiB 在有效 20Mbps 下约 35.8 分钟，100Mbps 下约 7.2 分钟；10GiB 分别约 71.6 / 14.3 分钟。解压、依赖、权重下载和首个 batch 排错另预留 20—60 分钟。已有云镜像与云盘数据时这部分可以大幅缩短。

若假设 S=3，纯计算节省是本地时长的 2/3；只有迁移开销低于 `T_local×(1−1/S)` 才缩短总等待。以数据流水线外推 {pipeline_hours:.2f} 小时计，这个阈值约 **{pipeline_hours*60*2/3:.0f} 分钟**。因此已有环境和数据的云卡可能省时，但不是基础任务能否完成的必要条件。

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
| 2D-FAP 2 epoch → 保存 → 新实例加载 | exact_equal={verification['2dfap']['exact_equal']}，最大差 {verification['2dfap']['max_abs_error']} |
| 年龄 2 epoch → 保存 → 新实例加载 | exact_equal={verification['appa']['exact_equal']}，最大差 {verification['appa']['max_abs_error']} |
| Windows DataLoader 多进程 | 两模型均 {verification['2dfap']['workers']} workers 干跑通过 |
| 分性别统计 | 合成 8 训练 / 4 留出贯通，未把不足样本的趋势标为可展示 |
| 五折汇总算术 | 合成预测的均值、样本标准差、pooled RMSE 与解析公式一致 |
| 原生 MediaPipe 提取入口 | 固定 v1 task、CPU 执行，12 张随机噪声全部记录为无脸 |
| ImageNet 初始化 | 两骨干下载并 strict 加载成功；SENet 镜像完整 SHA256 已核对 |

迁移对照使用现代 PyTorch 中运行的官方源码，只改废弃 import；没有另建 PyTorch 1.8 环境。权重全部为 tensor-only `{{"state_dict": ...}}`，未序列化模型对象，加载 `weights_only=True` 与 `strict=True`。

交付脚本、数据结构、单折/五折训练、数据核查、几何 JSON 字段、索引表与后端权重加载方式详见 [training/README.md](../training/README.md)。正式数据就位后先运行 `check_datasets.py`，再开训；正式五折指标和真实相关性将由脚本产出，不由这份测速报告填写。
'''
    path = ROOT.parent/'docs/TRAINING_ASSESSMENT.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
