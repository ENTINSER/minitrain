"""LoRA configuration builder for MiniTrain."""

from peft import LoraConfig, TaskType


# Preset target module mapping for supported model families.
TARGET_MODULES_MAP = {
    "qwen2.5": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _guess_target_modules(base_model_name: str) -> list[str]:
    """Infer LoRA target modules from the base model name."""
    lower = base_model_name.lower()
    for key, modules in TARGET_MODULES_MAP.items():
        if key in lower:
            return modules
    # Default to the Qwen2.5 preset, which is the primary supported model family.
    return TARGET_MODULES_MAP["qwen2.5"]


def get_lora_config(base_model_name: str, config: dict) -> LoraConfig:
    """Build a ``peft.LoraConfig`` from the global config dictionary.

    Args:
        base_model_name: Hugging Face model id or local path for the base model.
        config: Parsed ``config.yaml`` dictionary. LoRA hyper-parameters are read
            from ``model.lora_r``, ``model.lora_alpha`` and ``model.lora_dropout``.

    Returns:
        A configured ``peft.LoraConfig`` instance.
    """
    model_cfg = config.get("model", {})

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        target_modules=_guess_target_modules(base_model_name),
        bias="none",
    )
