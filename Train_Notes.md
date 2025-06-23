## --model_name_or_path / --model-name-or-path
- **类型**：str
- **默认值**：无
- **说明**：模型路径或名称（支持本地路径或 HuggingFace 上的模型 ID）

## --adapter_name_or_path / --adapter-name-or-path
- **类型**：str
- **默认值**：无
- **说明**：LoRA adapter 的路径或名称，支持多个，用逗号分隔

## --adapter_folder / --adapter-folder
- **类型**：str
- **默认值**：无
- **说明**：包含 adapter 权重文件的文件夹路径

## --cache_dir / --cache-dir
- **类型**：str
- **默认值**：无
- **说明**：用于缓存模型和 tokenizer 的目录

## --use_fast_tokenizer / --use-fast-tokenizer  
- **类型**：bool  
- **可接受值**：`true`, `false`  
- **默认值**：`true`  
- **说明**：是否启用 fast tokenizer（基于 `tokenizers` 库）

## --no_use_fast_tokenizer / --no-use-fast-tokenizer  
- **类型**：bool  
- **说明**：等效于 `--use_fast_tokenizer false`，用于显式关闭 fast tokenizer

## --resize_vocab / --resize-vocab
- **类型**：bool
- **默认值**：false
- **说明**：是否根据新 tokenizer 大小调整词表和 embedding 层大小

## --split_special_tokens / --split-special-tokens
- **类型**：bool
- **默认值**：false
- **说明**：是否在分词时分割特殊 token

## --add_tokens / --add-tokens
- **类型**：str（多个 token 用逗号分隔）
- **默认值**：无
- **说明**：要添加的非特殊 token

## --add_special_tokens / --add-special-tokens
- **类型**：str（多个 token 用逗号分隔）
- **默认值**：无
- **说明**：要添加的特殊 token

## --model_revision / --model-revision
- **类型**：str
- **默认值**：`main`
- **说明**：HuggingFace 模型的分支名、tag 或 commit ID

## --low_cpu_mem_usage / --low-cpu-mem-usage
- **类型**：bool
- **默认值**：true
- **说明**：是否使用低内存模式加载模型

## --no_low_cpu_mem_usage / --no-low-cpu-mem-usage
- **类型**：bool
- **说明**：等效于设置 `--low_cpu_mem_usage false`

## --rope_scaling / --rope-scaling
- **类型**：enum
- **可接受值**：`linear`, `dynamic`, `yarn`, `llama3`
- **默认值**：无
- **说明**：RoPE（旋转位置编码）扩展方式，适用于长上下文场景

## --flash_attn / --flash-attn
- **类型**：enum
- **可接受值**：`auto`, `disabled`, `sdpa`, `fa2`
- **默认值**：`auto`
- **说明**：是否使用 FlashAttention 或其他注意力加速模块

## --shift_attn / --shift-attn
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 Shift Attention（来自 LongLoRA）

## --mixture_of_depths / --mixture-of-depths
- **类型**：enum
- **可接受值**：`convert`, `load`
- **默认值**：无
- **说明**：使用 MoD（Mixture-of-Depths）时选择转换已有模型或加载已有 MoD 模型

## --use_unsloth / --use-unsloth
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 [Unsloth](https://github.com/unslothai/unsloth) 的优化加速（更快的 LoRA 训练）

## --use_unsloth_gc / --use-unsloth-gc
- **类型**：bool
- **默认值**：false
- **说明**：是否使用 Unsloth 提供的 gradient checkpointing 方法

## --enable_liger_kernel / --enable-liger-kernel
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 Liger 内核优化，加速推理或训练

## --moe_aux_loss_coef / --moe-aux-loss-coef
- **类型**：float
- **默认值**：无
- **说明**：混合专家模型中 auxiliary router loss 的权重系数

## --disable_gradient_checkpointing / --disable-gradient-checkpointing
- **类型**：bool
- **默认值**：false
- **说明**：是否禁用梯度检查点（通常用于节省内存）

## --use_reentrant_gc / --use-reentrant-gc
- **类型**：bool
- **默认值**：true
- **说明**：是否使用 PyTorch 的 reentrant gradient checkpointing 实现

## --no_use_reentrant_gc / --no-use-reentrant-gc
- **类型**：bool
- **说明**：等效于设置 `--use_reentrant_gc false`

## --upcast_layernorm / --upcast-layernorm
- **类型**：bool
- **默认值**：false
- **说明**：是否将 LayerNorm 权重上采样为 float32（可改善数值稳定性）

## --upcast_lmhead_output / --upcast-lmhead-output
- **类型**：bool
- **默认值**：false
- **说明**：是否将 `lm_head` 输出结果上采样为 float32

## --train_from_scratch / --train-from-scratch
- **类型**：bool
- **默认值**：false
- **说明**：是否从头训练模型（不加载预训练权重）

## --infer_backend / --infer-backend
- **类型**：enum
- **可接受值**：`hf`, `vllm`, `ctransformers`, `llama-cpp`, `mlc`
- **默认值**：`hf`
- **说明**：推理使用的后端框架，可选择 HuggingFace 或高性能引擎如 vLLM、ctransformers 等

## --use_cache / --use-cache
- **类型**：bool
- **默认值**：true
- **说明**：训练或推理过程中是否使用 attention cache 机制

## --no_use_cache / --no-use-cache
- **类型**：bool
- **说明**：等效于 `--use_cache false`

## --quantization_bit / --quantization-bit
- **类型**：int
- **可接受值**：`4`, `8`
- **默认值**：无
- **说明**：指定量化精度（bit），例如 INT4 或 INT8 量化权重

## --quantization_method / --quantization-method
- **类型**：enum
- **可接受值**：`gptq`, `awq`, `bnb`, `cpm`, `mlc`
- **默认值**：无
- **说明**：量化方法的类型（支持 GPTQ、AWQ、bitsandbytes、CPM、MLC 等）

## --use_triton / --use-triton
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 Triton 加速推理（适用于 bitsandbytes）

## --use_exllama / --use-exllama
- **类型**：bool
- **默认值**：false
- **说明**：是否使用 exllama 推理后端（配合 INT4）

## --strict_dtype / --strict-dtype
- **类型**：bool
- **默认值**：false
- **说明**：是否严格使用配置中指定的权重数据类型

## --force_auto_device / --force-auto-device
- **类型**：bool
- **默认值**：false
- **说明**：强制使用自动设备映射（用于不支持部分 dtype 的模型）

## --use_safetensors / --use-safetensors
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 safetensors 格式加载权重（更安全、加载更快）

## --trust_remote_code / --trust-remote-code
- **类型**：bool
- **默认值**：false
- **说明**：是否信任 HuggingFace 上模型提供的远程代码（可能有安全风险）

## --attn_implementation / --attn-implementation
- **类型**：enum
- **可接受值**：`eager`, `sdpa`, `flash_attention_2`
- **默认值**：`eager`
- **说明**：选择 attention 实现方式（手写、SDPA 或 FlashAttention 2）

## --finetuning_type / --finetuning-type
- **类型**：enum
- **可接受值**：`none`, `full`, `freeze`, `lora`
- **默认值**：`none`
- **说明**：微调策略类型：
  - `none`：不微调
  - `full`：全参数微调
  - `freeze`：冻结部分参数，仅训练头部
  - `lora`：使用 LoRA 参数高效微调

## --lora_rank / --lora-rank
- **类型**：int
- **默认值**：8
- **说明**：LoRA 的秩（低秩矩阵的维度），控制参数量和表达能力

## --lora_alpha / --lora-alpha
- **类型**：float
- **默认值**：32
- **说明**：LoRA 中的缩放因子（与 rank 联合影响更新幅度）

## --lora_dropout / --lora-dropout
- **类型**：float
- **默认值**：0.05
- **说明**：LoRA 中的 dropout 比例，用于正则化训练

## --lora_target / --lora-target
- **类型**：str 或 逗号分隔列表
- **默认值**：无
- **说明**：指定注入 LoRA 的目标模块名（如 `q_proj`, `v_proj`, `all`）

## --lora_weight_path / --lora-weight-path
- **类型**：str
- **默认值**：无
- **说明**：已训练好的 LoRA 权重路径（用于继续训练或推理）

## --lora_bias / --lora-bias
- **类型**：enum
- **可接受值**：`none`, `all`, `lora_only`
- **默认值**：`none`
- **说明**：
  - `none`：不训练 bias
  - `all`：训练所有 bias
  - `lora_only`：仅训练 LoRA 层相关 bias

## --merge_lora / --merge-lora
- **类型**：bool
- **默认值**：false
- **说明**：是否将 LoRA 权重合并入原模型（适用于部署或导出）

## --merge_lora_and_save / --merge-lora-and-save
- **类型**：bool
- **默认值**：false
- **说明**：是否在保存时直接合并 LoRA 并保存整合模型

## --tuner_backend / --tuner-backend
- **类型**：enum
- **可接受值**：`peft`, `hqq`
- **默认值**：`peft`
- **说明**：指定微调后端库，支持 Hugging Face PEFT 或 HQQ

## --plot_lora_matrix / --plot-lora-matrix
- **类型**：bool
- **默认值**：false
- **说明**：是否在训练后绘制 LoRA 参数矩阵的热图（用于可视化分析）

## --lora_fan_in_fan_out / --lora-fan-in-fan-out
- **类型**：bool
- **默认值**：false
- **说明**：是否切换 LoRA 的矩阵乘法顺序（针对部分模型结构）

## --quant_lora / --quant-lora
- **类型**：bool
- **默认值**：false
- **说明**：是否对 LoRA 参数进行量化（如 INT8），用于减少显存占用

## --dataset
- **类型**：str（多个数据集以逗号分隔）
- **默认值**：无
- **说明**：指定用于训练/评估的数据集名称或配置，如 `alpaca_zh`, `identity`, `sharegpt`

## --dataset_dir / --dataset-dir
- **类型**：str
- **默认值**：无
- **说明**：自定义数据集所在的文件夹路径（如本地 JSON 或 CSV 文件）

## --overwrite_cache / --overwrite-cache
- **类型**：bool
- **默认值**：false
- **说明**：是否覆盖缓存的数据处理结果（如已缓存 tokenizer 输出）

## --max_samples / --max-samples
- **类型**：int
- **默认值**：无（表示使用全部样本）
- **说明**：限制训练或评估时加载的样本数

## --preprocessing_num_workers / --preprocessing-num-workers
- **类型**：int
- **默认值**：null（通常为单线程）
- **说明**：数据预处理时使用的并行线程数（推荐根据 CPU 核心数设置）

## --cutoff_len / --cutoff-len
- **类型**：int
- **默认值**：512
- **说明**：输入序列最大 token 长度，超出将被截断

## --template
- **类型**：str
- **可接受值**：`auto`, `llama2`, `llama3`, `baichuan`, `chatml`, `qwen`, `internlm`, `custom`, ...
- **默认值**：`auto`
- **说明**：用于格式化输入的 Prompt 模板，指定不同模型的 prompt 格式要求

## --template_file / --template-file
- **类型**：str
- **默认值**：无
- **说明**：自定义模板文件路径（用于非预定义格式）

## --template_type / --template-type
- **类型**：enum
- **可接受值**：`default`, `chat`, `completion`
- **默认值**：`default`
- **说明**：模板的输出类型：completion（无 assistant），chat（包含对话格式）

## --prompt_column / --prompt-column
- **类型**：str
- **默认值**：`instruction`
- **说明**：数据集中用于生成 prompt 的字段名称

## --query_column / --query-column
- **类型**：str
- **默认值**：`input`
- **说明**：用于问句（用户输入）的字段名称

## --response_column / --response-column
- **类型**：str
- **默认值**：`output`
- **说明**：用于答案/响应的字段名称

## --history_column / --history-column
- **类型**：str
- **默认值**：`history`
- **说明**：多轮对话中用于历史上下文的字段名称

## --learning_rate / --learning-rate
- **类型**：float
- **默认值**：`5e-5`
- **说明**：初始学习率

## --num_train_epochs / --num-train-epochs
- **类型**：float 或 int
- **默认值**：`3.0`
- **说明**：总训练轮数

## --max_steps / --max-steps
- **类型**：int
- **默认值**：`-1`（表示根据 epoch 计算）
- **说明**：总训练步数；若设置为正整数，将覆盖 `num_train_epochs`

## --per_device_train_batch_size / --per-device-train-batch-size
- **类型**：int
- **默认值**：`8`
- **说明**：每个设备（GPU）上的训练 batch size

## --per_device_eval_batch_size / --per-device-eval-batch-size
- **类型**：int
- **默认值**：`8`
- **说明**：每个设备上的验证 batch size

## --gradient_accumulation_steps / --gradient-accumulation-steps
- **类型**：int
- **默认值**：`1`
- **说明**：梯度累积步数，等效于增大实际 batch size

## --lr_scheduler_type / --lr-scheduler-type
- **类型**：enum
- **可接受值**：`linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`
- **默认值**：`cosine`
- **说明**：学习率调度策略

## --warmup_steps / --warmup-steps
- **类型**：int
- **默认值**：`0`
- **说明**：学习率预热步数

## --warmup_ratio / --warmup-ratio
- **类型**：float（0 到 1）
- **默认值**：`0.03`
- **说明**：预热步数比例，优先于 `warmup_steps` 生效

## --weight_decay / --weight-decay
- **类型**：float
- **默认值**：`0.0`
- **说明**：权重衰减系数（L2 正则）

## --adam_beta1 / --adam-beta1
- **类型**：float
- **默认值**：`0.9`
- **说明**：Adam 优化器 beta1 系数

## --adam_beta2 / --adam-beta2
- **类型**：float
- **默认值**：`0.999`
- **说明**：Adam 优化器 beta2 系数

## --adam_epsilon / --adam-epsilon
- **类型**：float
- **默认值**：`1e-8`
- **说明**：Adam 优化器的 epsilon 值

## --max_grad_norm / --max-grad-norm
- **类型**：float
- **默认值**：`1.0`
- **说明**：最大梯度裁剪阈值，防止梯度爆炸

## --gradient_checkpointing / --gradient-checkpointing
- **类型**：bool
- **默认值**：false
- **说明**：是否启用梯度检查点节省显存（以牺牲计算换内存）

## --ddp_find_unused_parameters / --ddp-find-unused-parameters
- **类型**：bool
- **默认值**：false
- **说明**：多卡分布式训练中，是否允许存在未参与梯度更新的参数

## --bf16
- **类型**：bool
- **默认值**：false
- **说明**：是否使用 bfloat16 训练精度（若硬件支持）

## --fp16
- **类型**：bool
- **默认值**：false
- **说明**：是否使用 float16 精度训练（若硬件支持）

## --fp32
- **类型**：bool
- **默认值**：false
- **说明**：强制使用 float32 精度训练

## --logging_steps / --logging-steps
- **类型**：int
- **默认值**：`10`
- **说明**：每隔多少步打印一次日志（如 loss）

## --save_steps / --save-steps
- **类型**：int
- **默认值**：`500`
- **说明**：每隔多少训练步保存一次 checkpoint

## --save_total_limit / --save-total-limit
- **类型**：int
- **默认值**：`3`
- **说明**：最多保存多少个模型 checkpoint，超过数量将覆盖旧的

## --save_safetensors / --save-safetensors
- **类型**：bool
- **默认值**：false
- **说明**：保存模型时是否使用 `.safetensors` 格式（更安全，加载更快）

## --output_dir / --output-dir
- **类型**：str
- **默认值**：无
- **说明**：模型及日志输出的目标路径

## --overwrite_output_dir / --overwrite-output-dir
- **类型**：bool
- **默认值**：false
- **说明**：如果输出目录存在，是否覆盖其内容

## --plot_loss / --plot-loss
- **类型**：bool
- **默认值**：false
- **说明**：是否在训练过程中记录并保存 loss 曲线图

## --logging_dir / --logging-dir
- **类型**：str
- **默认值**：`runs`
- **说明**：TensorBoard 等日志的保存目录

## --disable_tqdm / --disable-tqdm
- **类型**：bool
- **默认值**：false
- **说明**：是否关闭 tqdm 进度条输出（适用于日志平台）

## --report_to / --report-to
- **类型**：str
- **可接受值**：`none`, `tensorboard`, `wandb`, `all`
- **默认值**：`none`
- **说明**：将训练指标报告到哪些工具或平台

## --run_name / --run-name
- **类型**：str
- **默认值**：无
- **说明**：训练运行的标识名称（如用于 wandb 或 tensorboard）

## --dataloader_num_workers / --dataloader-num-workers
- **类型**：int
- **默认值**：0
- **说明**：数据加载时使用的线程数量，通常设为 CPU 核心数的一部分

## --dataloader_pin_memory / --dataloader-pin-memory
- **类型**：bool
- **默认值**：true
- **说明**：是否将数据固定到内存中（提升数据加载性能）

## --ddp_timeout / --ddp-timeout
- **类型**：int
- **默认值**：`1800`（秒）
- **说明**：分布式训练中，初始化阶段的最大等待时间

## --evaluation_strategy / --evaluation-strategy
- **类型**：enum
- **可接受值**：`no`, `steps`, `epoch`
- **默认值**：`no`
- **说明**：指定验证触发方式：
  - `no`：不进行评估
  - `steps`：每隔一定训练步数评估
  - `epoch`：每个 epoch 结束后评估

## --eval_steps / --eval-steps
- **类型**：int
- **默认值**：`500`
- **说明**：每隔多少步评估一次（仅在 `evaluation_strategy=steps` 时生效）

## --val_size / --val-size
- **类型**：float 或 int
- **默认值**：`0.1`
- **说明**：从训练集中划分出的验证集比例（如 0.1 表示 10%）

## --load_best_model_at_end / --load-best-model-at-end
- **类型**：bool
- **默认值**：false
- **说明**：训练结束后是否加载验证指标最优的模型 checkpoint

## --metric_for_best_model / --metric-for-best-model
- **类型**：str
- **默认值**：`loss`
- **说明**：用于判断最佳模型的评估指标名称（如 `accuracy`, `loss`）

## --greater_is_better / --greater-is-better
- **类型**：bool
- **默认值**：false
- **说明**：是否认为指标越大越好（如用于 `accuracy`）；若评估的是 `loss`，应设为 `false`

## --early_stopping_patience / --early-stopping-patience
- **类型**：int
- **默认值**：`3`
- **说明**：在指标无提升的情况下，最多允许的评估次数（超过后中止训练）

## --early_stopping_threshold / --early-stopping-threshold
- **类型**：float
- **默认值**：`0.0`
- **说明**：指标变化必须超过该阈值才视为“有提升”

## --export_dir / --export-dir
- **类型**：str
- **默认值**：无
- **说明**：导出最终模型的目标目录（可与 `--merge_lora_and_save` 联合使用）

## --export_onnx / --export-onnx
- **类型**：bool
- **默认值**：false
- **说明**：是否导出为 ONNX 格式模型（适用于部署和跨平台推理）

## --merge_lora / --merge-lora
- **类型**：bool
- **默认值**：false
- **说明**：是否在推理前将 LoRA 权重合并到原始模型中

## --merge_lora_and_save / --merge-lora-and-save
- **类型**：bool
- **默认值**：false
- **说明**：是否将合并后的模型权重保存为一个完整模型（不再依赖 LoRA adapter）

## --quant_lora / --quant-lora
- **类型**：bool
- **默认值**：false
- **说明**：是否对 LoRA adapter 本身进行量化（用于部署节省资源）

## --save_safetensors / --save-safetensors
- **类型**：bool
- **默认值**：false
- **说明**：保存模型时是否使用 `.safetensors` 格式（更安全，加载更快）

## --trust_remote_code / --trust-remote-code
- **类型**：bool
- **默认值**：false
- **说明**：加载模型时是否信任第三方自定义的 `modeling.py`（慎用）

## --auto_merge_weights / --auto-merge-weights
- **类型**：bool
- **默认值**：false
- **说明**：是否自动合并多个 adapter 权重（用于多 LoRA 组合推理）

## --deepspeed
- **类型**：str
- **默认值**：无
- **说明**：指定 DeepSpeed 配置文件路径（如 `ds_config.json`），启用 DeepSpeed 加速训练

## --ddp_timeout / --ddp-timeout
- **类型**：int
- **默认值**：1800
- **说明**：DDP 初始化等待时间（秒），防止部分节点初始化过慢导致失败

## --ddp_find_unused_parameters / --ddp-find-unused-parameters
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 DDP 中未使用参数的检测（适用于多分支结构）

## --fsdp
- **类型**：str
- **默认值**：无
- **说明**：Fully Sharded Data Parallel（FSDP）配置，例如 `'full_shard auto_wrap'`

## --fsdp_config / --fsdp-config
- **类型**：str
- **默认值**：无
- **说明**：FSDP 配置文件路径

## --bf16
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 bfloat16 精度训练（要求 GPU 支持，如 A100/H100）

## --fp16
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 float16 精度训练（兼容性广）

## --fp32
- **类型**：bool
- **默认值**：false
- **说明**：是否强制使用 float32 精度（用于关闭混合精度）

## --tf32
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 TensorFloat-32（NVIDIA Ampere 架构支持）

## --no_cuda / --no-cuda
- **类型**：bool
- **默认值**：false
- **说明**：是否禁用 GPU 训练，仅使用 CPU

## --device_map / --device-map
- **类型**：str
- **默认值**：`auto`
- **说明**：模型在多 GPU 上的映射方式，如 `auto`, `balanced`, `sequential`

## --tpu_num_cores / --tpu-num-cores
- **类型**：int
- **默认值**：无
- **说明**：指定使用的 TPU 核心数（适用于 Google Cloud TPU）

## --tpu_metrics_debug / --tpu-metrics-debug
- **类型**：bool
- **默认值**：false
- **说明**：是否开启 TPU 训练的调试信息输出

## --torch_compile / --torch-compile
- **类型**：bool
- **默认值**：false
- **说明**：是否启用 PyTorch 2.0 的 torch.compile 模型优化功能

## --use_reentrant_gc / --use-reentrant-gc
- **类型**：bool
- **默认值**：true
- **说明**：是否启用 reentrant gradient checkpointing 模式（更节省显存）

## --gradient_checkpointing / --gradient-checkpointing
- **类型**：bool
- **默认值**：false
- **说明**：是否开启梯度检查点（减少显存，提升训练稳定性）

