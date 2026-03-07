from asymkv.method.asymkv_forward.asymkv_mistral import (
    enable_mistral_asymkv_attention,
)
from asymkv.method.asymkv_forward.asymkv_llama import (
    enable_llama_pos_shift_asymkv_attention_442,
)
from asymkv.method.asymkv_forward.asymkv_qwen2 import (
    enable_qwen2_asymkv_attention,
)
from asymkv.method.asymkv_forward.asymkv_gemma import (
    enable_gemma_asymkv_attention,
)


def enable_kvslimmer_attention(model_name, model):
    """
    Enable AsymKV-style attention patch according to model family.
    """
    name = model_name.lower()

    if "llama" in name:
        enable_llama_pos_shift_asymkv_attention_442(model)
    elif "mistral" in name:
        enable_mistral_asymkv_attention(model)
    elif "qwen" in name:
        enable_qwen2_asymkv_attention(model)
    elif "gemma" in name:
        enable_gemma_asymkv_attention(model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")