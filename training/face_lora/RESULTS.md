# InternVL3-2B 五官描述训练结果

2026 年 11 月 18 日，完成一次小规模验证和一轮正式训练。正式权重已回迁。v1.2 将它接入默认描述后端。现有评分继续使用 2D-FAP。

## 训练与交付

| 项目 | 实际配置或结果 |
| --- | --- |
| 基模 | OpenGVLab/InternVL3-2B-hf，提交 `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` |
| 设备与输入 | RTX PRO 6000 Blackwell Server Edition，BF16，每脸 1 个 448×448 图块 |
| 可训练部分 | 语言模型注意力 Q/K/V/O，112 个模块，LoRA rank 16、alpha 32、dropout 0.05，共 4,358,144 个参数 |
| 小试 | 384 个训练样本、48 个开发样本，24 步。开发损失 3.153 → 1.529 |
| 正式训练 | SCUT 几何弱监督 3,609 张、Face2Text 人工描述配图 1,499 张、UniF²aceD 自动描述配图 1,500 张，共 6,608 张。批量 16，一轮 413 步，初始学习率 0.0001 |
| 开发集 | SCUT 375 张，生成目标的 token 损失 3.156 → 0.0467 |
| 适配器 | `backend/weights/InternVL3-2B-face-lora/adapter_model.safetensors`，17,465,792 字节，SHA-256 为 `714144a59af9b8ecca4841bce6bc19efdd7101a5a90589df85dc4c27ce659a1e` |

SCUT 的训练目标来自关键点几何和可见动作。Face2Text 使用英文人工描述中可明确提取的属性，UniF²aceD 使用生成描述中的局部属性。UniF²aceD 的 MM-CelebA-HQ 图号通过 30,000 行映射与同一张 CelebA 对齐图配对。两套外部语料均按单部位短问答训练。正式训练纳入外部留出图哈希进行排重。原有六张回归照片也列入排除清单。[训练清单及损失记录](../../training/artifacts/face_lora/remote_20261118/runs/formal/training_stats.json)

## 视觉核查

公开留出集共 224 张，按来源和部位分层取 48 张，对比原版 2B 和 LoRA。两者均有 48 条可解析回答。严格裸 JSON 从 0 条增至 48 条，预设词库中的正属性命中从 7 条增至 15 条。词库识别了基模 19 条、LoRA 48 条回答，逐条检查还能在基模输出中找到至少 3 条被漏计的同义描述。[公开留出集逐条生成](../../training/artifacts/face_lora/remote_20261118/public_ab/public_eval.jsonl)

选取 7 对有相反属性标注的照片做双向换图。两版均未在同一对图上同时说对两侧属性并随图切换。LoRA 在 14 次换图中有 6 次改变回答。六张回归照上，LoRA 把三张露齿微笑照写成嘴角上扬，在另外三张写出嘴唇合拢。但六张的眉形句和轮廓句都分别完全相同，五张的鼻翼宽度描述也相同。[换图结果](../../training/artifacts/face_lora/remote_20261118/public_ab/summary.json) · [六张回归照原文](../../training/artifacts/face_lora/remote_20261118/six_lora.jsonl)

本地 RTX 4060 对同一批 40 张冻结裁剪完成三模型对照。LoRA 40/40 次生成严格六键 JSON，240/240 个字段均给出可见描述，文字生成中位耗时 4.65 秒。两种基模的输出均可经现有解析器读取，严格裸 JSON 均为 0/40。[本地三模型指标](../artifacts/face_lora_eval/heldout40/summary.json)

本地冻结的 12 张视觉核查中，按应用最终可见文字检查 23 个显见事实，FaceLLM-1B、原版 2B、LoRA 分别支持 13、14、13 项。LoRA 的明确冲突或遮挡时仍断言有 3 项，未覆盖 7 项。该比较记录了迁移前的实验表现，v1.2 默认使用 LoRA。[核查明细](../../training/artifacts/face_lora_eval/heldout40/visual_audit.md) · [后端安装方法](../../backend/README.md)

本地真实单人请求依次返回 `scores_ready` 和 `completed`。RTX 4060 同时加载 FaceLLM 门禁与 LoRA 描述后，预热峰值显存为 5.82 GiB，该请求总耗时 5.53 秒。远程训练实例于北京时间 12:26:57 执行平台关机命令并返回成功，随后三次连接均被拒绝。[本地 API 实测](../artifacts/face_lora_eval/heldout40/lora_live.json)

## 辅助评分

在 SCUT 的 5,500 张图上，原始评分、仅用分数校准、再加入 15 项几何属性的合并 MAE 分别为 0.215196、0.213993、0.212887。几何相对仅校准分数的配对差值为 −0.001107，95% 区间为 [−0.002336, +0.000178]。这组结果支持继续保留现有数字评分。该项评测输入是关键点几何，LoRA 自由文本对评分的增量价值尚未实测。[五折方法与完整结果](../face_calibration/README.md)
