# coding=utf-8
"""Qwen3 model configuration for vLLM 0.6.3 (backport).

transformers 4.46.3 has no Qwen3Config; HF checkpoints use model_type=qwen3.
We subclass Qwen2Config so existing fields keep working, and add head_dim.
"""
from transformers import Qwen2Config


class Qwen3Config(Qwen2Config):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, head_dim=None, attention_bias=False, **kwargs):
        # Qwen3 stores head_dim explicitly; it is often NOT
        # hidden_size // num_attention_heads (e.g. 2560/32=80 but head_dim=128).
        super().__init__(**kwargs)
        if head_dim is not None:
            self.head_dim = head_dim
        elif not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # HF Qwen3 uses attention_bias (default False); keep attribute visible.
        self.attention_bias = attention_bias
