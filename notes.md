# SFT+LoRA 微调配置参数说明文档

## 一、基本配置

| 参数名 | 可接受的输入值 | 说明 |
|--------|----------------|------|
| `model_name_or_path` | 任意 HuggingFace 模型名或本地路径<br>如：`meta-llama/Meta-Llama-3-8B-Instruct`, `./models/llama3` | 指定用于微调的基础模型。 |
| `stage` | `pt`, `sft`, `rm`, `ppo`, `dpo`, `kto` | 指定微调所处阶段：<br>• `pt`：pretraining <br>• `sft`：Supervised Fine-Tuning（常见）<br>• `rm`：Reward Model<br>• `ppo`：Proximal Policy Optimization（RLHF）<br>• `dpo`：Direct Preference Optimization <br>• `kto`：Knowledge Transfer Optimization |
| `do_train` | `true`, `false` | 是否执行训练流程。<br>设为 `true` 时将执行训练主循环。<br>**调试建议**：可设为 `false` 只进行推理或验证。 |
| `do_predict` | `true`, `false` | 是否进行推理预测。<br>**注意**：若要在训练后自动推理生成结果，需设为 `true`。 |
| `finetuning_type` | `lora`, `full`, `freeze` | 指定微调策略：<br>• `lora`：参数高效微调，仅训练少量 adapter（推荐）<br>• `full`：全量微调，显存消耗大<br>• `freeze`：冻结部分参数，仅微调特定模块（需额外配置） |
| `lora_target` | `all`, `q_proj`, `v_proj`, `gate_proj`, 逗号分隔模块名 | 指定哪些模块注入 LoRA。<br>• `all` 表示自动注入支持的所有线性层<br>• 自定义设置可选 `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` 等<br>**Tips**：仅选择 `q_proj,v_proj` 可减少参数量并提升推理效率。 |

---

## 二、数据与预处理

| 参数名 | 可接受的输入值 | 说明 |
|-|-|-|
| `dataset` | 数据集名称字符串，多个以`,`分隔（如 `identity,alpaca_en_demo`） | 指定训练所用的数据集名称。<br>**说明**：支持 `Alpaca` 格式和 `ShareGPT` 格式的数据集。 |
| `template` | `llama2`, `llama3`, `chatml`, `baichuan`, `qwen`, `custom` 等 | 选择用于构建 prompt 的模板。<br>**建议**：必须与预训练模型的 prompt 格式匹配。例如 `llama3` 要配合 LLaMA 3 使用。若使用自定义格式，请选择 `custom` 并指定模板文件。 |
| `cutoff_len` | 正整数（如 `512`, `1024`, `2048`） | 单条样本输入文本的最大 token 长度，超出部分将被截断。<br>**建议**：设置不应超过模型的最大上下文（如 LLaMA 2 是 4096），LoRA 场景下通常使用 `512`~`1024`。 |
| `max_samples` | 正整数，或默认值表示不限制 | 限定用于训练的数据样本数，用于调试或小规模测试。<br>**说明**：设置为 `1000` 可用于快速过拟合测试；生产训练推荐 `-1`（全量）。 |
| `overwrite_cache` | `true`, `false` | 是否重新处理并覆盖之前的数据缓存。<br>**建议**：修改模板、tokenizer 或数据内容后应设为 `true`；否则设为 `false` 以节省预处理时间。 |
| `preprocessing_num_workers` | 正整数（如 `4`, `8`, `16`） | 数据预处理并行进程数量，提升数据加载速度。 |

---

## 三、输出与日志设置

| 参数名 | 可接受的输入值 | 说明 |
|-|-|-|
| `output_dir` | 有效的路径字符串（如 `./saves/llama3-8b/lora/sft`） | 模型输出目录，保存训练结果、checkpoint 和日志等内容。<br>**建议**：使用区分模型和任务的子目录，便于后续管理和复现。 |
| `logging_steps` | 正整数（如 `10`, `50`, `100`） | 每隔多少训练步打印一次日志（如 loss、learning rate）。<br>**影响**：值越小观察越细但影响性能，推荐设置为 `10`~`100`。 |
| `save_steps` | 正整数（如 `500`, `1000`, `2000`） | 每隔多少训练步保存一次模型 checkpoint。<br>**建议**：保存间隔视数据量、实验稳定性和存储空间决定。较小值便于恢复，较大值减少磁盘占用。 |
| `plot_loss` | `true`, `false` | 是否在训练结束后生成并保存 loss 曲线图（通常保存在 output_dir 下的 `trainer_state.json` 旁）。<br>**建议**：调参阶段启用以便可视化 loss 趋势。 |
| `overwrite_output_dir` | `true`, `false` | 如果输出目录已存在，是否清空其中内容并重新写入。<br>**注意**：设为 `true` 会删除已有 checkpoint，确保不是误覆盖；设为 `false` 时若目录非空将报错。建议训练正式版本前启用。 |

---

## 四、训练参数

| 参数名 | 可接受的输入值 | 说明 |
|-|-|-|
| `per_device_train_batch_size` | 正整数（如 `1`, `4`, `8`, `16`） | 每个设备上的训练 batch size。<br>**影响**：值越大训练越快，但显存占用也更高；值越小更省显存但训练波动大。 |
| `gradient_accumulation_steps` | 正整数（如 `1`, `4`, `8`, `16`） | 梯度累积步数。<br>**影响**：模拟更大 batch 大小，无需额外显存。推荐设置为 `4`~`16` 以提升稳定性，尤其是在 batch size 小的设备上。 |
| `learning_rate` | 正浮点数（如 `1e-4`, `5e-5`, `2e-5`） | 初始学习率。<br>**建议**：LoRA 微调一般使用 `1e-4` ~ `2e-5`，full finetune 推荐 `5e-5` ~ `1e-5`。过大可能 loss 不收敛，过小则收敛慢。 |
| `num_train_epochs` | 正整数或浮点数（如 `3`, `5`, `1.5`） | 总训练轮数。<br>**影响**：轮数越多训练越充分，但风险过拟合。小数据集建议 `3`~`10`，大数据集常配合 early stopping 使用。 |
| `lr_scheduler_type` | `cosine`, `linear`, `constant`, `polynomial`, 等 | 学习率调度器。<br>• `cosine`：先快后慢，训练稳定<br>• `linear`：简单线性下降，适合基础任务<br>• `constant`：固定学习率，调试或短任务使用 |
| `warmup_ratio` | 0 ~ 1 之间的浮点数（如 `0.03`, `0.1`） | 学习率预热比例。<br>**影响**：预热阶段防止震荡，LoRA 推荐设置为 `0.03` 以上，提升 early training 稳定性。 |
| `bf16` | `true`, `false` | 是否启用 bfloat16 精度训练。<br>**说明**：可节省显存且更稳定（比 fp16 有更宽的数值范围），需硬件支持（如 A100、H100）。 |
| `ddp_timeout` | 正整数（单位：秒） | DDP 分布式训练启动最大等待时间。<br>**建议**：多 GPU 训练时设置为 `1800` 秒（默认）避免节点启动不同步导致失败。 |

---

## 五、验证配置

| 参数名 | 可接受的输入值 | 说明 |
|-|-|-|
| val_size | 0 到 1 之间的浮点数 | 验证集所占比例（如 `0.1`） |
| per_device_eval_batch_size | 正整数 | 每设备验证 batch size |
| eval_strategy | `no`, `steps`, `epoch` | 验证触发策略：不验证、按步、按 epoch |
| eval_steps | 正整数 | 每隔多少步验证一次（需配合 `eval_strategy: steps`） |

---

# LLM 微调阶段说明

| 阶段名 | 全称 | 作用 | 输入数据类型 |
|-|-|-|-|
| `pt`   | Pretraining | 无监督预训练，语言建模 | 大规模未标注文本 |
| `sft`  | Supervised Fine-Tuning | 有监督指令微调，模型学会基本任务 | 指令 + 输入 + 输出 |
| `rm`   | Reward Model Training | 奖励模型训练，学习“偏好排序” | 成对回答 + 偏好 |
| `ppo`  | Proximal Policy Optimization | 强化学习，用奖励模型优化行为 | Prompt + 奖励 |
| `dpo`  | Direct Preference Optimization | 无需奖励模型直接对比优化 | Chosen vs Rejected 回答 |
| `kto`  | Knowledge Transfer Optimization | 无需奖励模型，通过 KL 散度对齐学生模型 | 学生回答 + 老师回答 |

---

# LoRA vs QLoRA：高效微调方法对比

## 一、概述

| 特性 | LoRA | QLoRA |
|-|-|-|
| 微调方式 | 注入低秩矩阵 | 注入低秩矩阵 + 4-bit 量化原始模型 |
| 显存占用 | 中等（使用 full precision 权重） | 极低（使用 4-bit INT 量化模型） |
| 模型修改 | 原始模型保持不变 | 原始模型需量化 |
| 支持模型大小 | 中等~大（推荐 7B~13B） | 超大（最大至 65B） |
| 微调精度 | 高（bf16/fp32） | 接近（INT4 + LoRA 权重） |
| 推理方式 | 可合并权重或保持 adapter 模式 | 通常保持 adapter 模式 |

---

## 二、核心技术对比

### LoRA（Low-Rank Adaptation）

- **原理**：将大矩阵的微调转换为两个小矩阵（A 和 B），显著减少参数数量。
- **优势**：
  - 高效稳定
  - 易于部署（可合并权重）
- **局限**：
  - 仍需 full-precision 模型（显存占用较高）

### QLoRA（Quantized LoRA）

- **原理**：在 LoRA 的基础上，将原始模型量化为 4-bit，结合 `NF4` + double quant 技术进行训练。
- **优势**：
  - 适用于资源受限设备（如 24GB 显卡、Colab）
  - 可微调超大模型（LLaMA2-65B 等）
- **局限**：
  - 推理时不能合并 LoRA（因底座是量化模型）
  - 依赖更多工具（如 bitsandbytes）

---

## 三、使用场景

| 场景 | 推荐方案 | 理由 |
|-|-|-|
| 显卡资源充足，追求部署合并模型 | **LoRA** | 支持权重合并，推理更方便 |
| 低资源环境（如 Colab、消费级显卡） | **QLoRA** | 低显存支持，适合快速实验 |
| 百亿参数以上模型的微调需求 | **QLoRA** | 显著减少内存使用，训练可行 |
| 精度要求极高的生产场景 | **LoRA** | 使用原始模型精度，收敛更稳 |

---