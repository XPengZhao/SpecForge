"""Metadata-only target configuration for offline Qwen4-Exp distillation.

No target modeling code is required: the training process loads only frozen
embedding/head tensors, never the hybrid/MoE/PLE target backbone.
"""
from transformers import AutoConfig, PretrainedConfig


class Qwen4ExpOfflineConfig(PretrainedConfig):
    model_type = "qwen4_exp"

    def __init__(self, text_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = PretrainedConfig(**(text_config or {}))


def load_offline_target_config(path, **kwargs):
    payload, _ = PretrainedConfig.get_config_dict(path, **kwargs)
    if payload.get("model_type") != "qwen4_exp":
        return AutoConfig.from_pretrained(path, **kwargs)
    if not isinstance(payload.get("text_config"), dict):
        raise ValueError("qwen4_exp requires a nested text_config")
    text = payload['text_config']
    for key in ('hidden_size', 'vocab_size', 'num_hidden_layers', 'hc_count'):
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ValueError(f"Invalid qwen4_exp text_config.{key}")
    if payload.get('quantization_config') or text.get('quantization_config'):
        raise ValueError('Qwen3.8 offline training requires the unquantized BF16 checkpoint')
    # Read the frozen head tying convention from the checkpoint, not the
    # generic PretrainedConfig default (which is True).
    payload.setdefault('tie_word_embeddings', text.get('tie_word_embeddings', False))
    return Qwen4ExpOfflineConfig.from_dict(payload)
