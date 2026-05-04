from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch

from .utils import require_import


@dataclass
class GemmaBundle:
    model: Any
    processor: Any
    device: torch.device


def load_gemma(
    model_id: str,
    dtype: str = "auto",
    device_map: str = "auto",
) -> GemmaBundle:
    transformers = require_import("transformers", "transformers")
    model_dtype = _resolve_dtype(dtype)
    processor = _load_processor(model_id, transformers)
    model_kwargs: dict[str, Any] = {"dtype": model_dtype}
    manual_device = _manual_device_for_device_map(device_map)
    if _should_pass_device_map(device_map):
        model_kwargs["device_map"] = device_map
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception:
        image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if image_text_cls is None:
            raise
        model = image_text_cls.from_pretrained(model_id, **model_kwargs)
    model.eval()
    if manual_device is not None:
        model.to(manual_device)
    device = next(model.parameters()).device
    return GemmaBundle(model=model, processor=processor, device=device)


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def apply_gemma_chat_template(processor: Any, messages: list[dict[str, str]]) -> str:
    template = getattr(processor, "apply_chat_template", None)
    if template is None:
        return _fallback_prompt_from_messages(messages)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return template(messages, **kwargs)


def parse_gemma_response(processor: Any, response: str) -> Any | None:
    parser = getattr(processor, "parse_response", None)
    if parser is None:
        return None
    try:
        return parser(response)
    except Exception:
        return None


def get_pad_token_id(processor: Any) -> int | None:
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    return None


def get_output_embedding_weight(model: Any) -> torch.Tensor | None:
    getter = getattr(model, "get_output_embeddings", None)
    if getter is None:
        return None
    embeddings = getter()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return None
    return embeddings.weight


def _load_processor(model_id: str, transformers: Any) -> Any:
    auto_processor = getattr(transformers, "AutoProcessor", None)
    if auto_processor is not None:
        try:
            return auto_processor.from_pretrained(model_id)
        except Exception:
            pass
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        raise RuntimeError(f"Unable to load a processor or tokenizer for '{model_id}'.")
    return auto_tokenizer.from_pretrained(model_id)


def _fallback_prompt_from_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _should_pass_device_map(device_map: Any) -> bool:
    if device_map is None:
        return False
    if isinstance(device_map, str):
        if device_map == "auto":
            return _accelerate_available()
        if _manual_device_for_device_map(device_map) is not None:
            return False
    return True


def _manual_device_for_device_map(device_map: Any) -> torch.device | None:
    if not isinstance(device_map, str) or device_map == "auto":
        return None
    try:
        return torch.device(device_map)
    except (TypeError, RuntimeError):
        return None


def _accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None
