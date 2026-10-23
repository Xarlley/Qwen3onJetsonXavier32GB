# Qwen3 → vLLM 0.6.3 纯 Python 移植说明

配合主 [README](../README.md) 步骤 D。不绑定任何私人绝对路径；下文以 `~/src/vllm` 为根。

## 1. 设计原则

- **只改 Python**，不改 `.cu`，不必为移植本身跑 `build_ext`。
- API **对齐本地 `model_executor/models/qwen2.py`（0.6.3）**，不要直接复制 0.8+ 的 `VllmConfig` 风格。
- 架构行为对齐上游 vLLM ≥0.8.4 的 Qwen3：`head_dim`、无 bias（按 config）、Q/K RMSNorm。

## 2. Config：`transformers_utils/configs/qwen3.py`

- 可继承 `transformers.Qwen2Config`，设置 `model_type = "qwen3"`。
- **必须**保留 / 暴露 **`head_dim`**（Qwen3-4B = 128）。
- 保留 **`attention_bias`**（权重里多为 `false`）。

在 `config.py` 的 `_CONFIG_REGISTRY` 增加：`"qwen3": Qwen3Config`。

## 3. 模型：`model_executor/models/qwen3.py`

建议从本地 `qwen2.py` 复制再改类名与下列差异：

| 点 | Qwen2（0.6.3） | Qwen3 |
|----|----------------|-------|
| head 维度 | `hidden_size // num_heads` | **`config.head_dim`** |
| QKV bias | 常为 True | **`attention_bias`** |
| Q/K norm | 无 | **per-head RMSNorm，再 RoPE** |

类命名：`Qwen3Attention` / `Qwen3DecoderLayer` / `Qwen3Model` / `Qwen3ForCausalLM`。

在 `registry.py` 注册：

```python
"Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
```

## 4. `sampling_params.py` guided 补丁

`GuidedDecodingParams.from_optional`：若所有 guided 字段为空，应 **`return None`**，避免空对象触发 outlines。

## 5. 验证清单

见主 README §8.4。改完后**重启** `api_server`。

## 6. 量化与后端（提醒）

- GPTQ：可用。AWQ：sm_72 上不可用。
- Attention：XFormers；不要依赖 flash-attn 路径。
