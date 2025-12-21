"""Utilities for constrained generation (semantic ID structured output).

This module provides a builder to create a `prefix_allowed_tokens_fn` that can
be passed to HuggingFace `generate` so that after a separator marker (e.g.,
"Response:") the model is only allowed to pick tokens from predefined sets
per position, enforcing a structured semantic ID like `<a_x> <b_y> <c_z>`.

Usage example:

    from verl.utils.constrained_generation import load_indices_file, build_semantic_prefix_allowed_tokens_fn
    indices = load_indices_file(path)  # List[List[str]] or Dict[str, List[str]]
    prefix_fn = build_semantic_prefix_allowed_tokens_fn(tokenizer, indices, sep_text="Response:")
    model.generate(..., prefix_allowed_tokens_fn=prefix_fn)

The indices structure can be one of:
1. List[List[str]]: each inner list is one possible sequence variant, each element is a lexical token string.
2. Dict[Any, List[str]]: values are sequence variants (same as (1)); keys are ignored.

All variants must share the same target length (e.g., 3 for `<a_*> <b_*> <c_*>`). For each position, the allowed
token set is the union of that position across all variants.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Union
import json
import logging
import torch

logger = logging.getLogger(__name__)

IndicesType = Union[List[List[str]], Dict[str, List[str]]]


def load_indices_file(path: str) -> IndicesType:
    """Load indices (semantic variants) from a JSON file.

    The JSON file can contain either:
      - a list of lists: [["<a_1>", "<b_2>", "<c_3>"], [...], ...]
      - a dict whose values are such lists: {"0": ["<a_1>", ...], ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for v in data.values():
            if not isinstance(v, list):
                raise ValueError("Each value in indices dict must be a list of token strings")
        return data  # type: ignore
    if isinstance(data, list):
        for v in data:
            if not isinstance(v, list):
                raise ValueError("Indices list elements must themselves be lists of token strings")
        return data  # type: ignore
    raise ValueError("Unsupported indices JSON structure")


def _normalize_indices(indices: IndicesType) -> List[List[str]]:
    if isinstance(indices, dict):
        variants = list(indices.values())
    else:
        variants = indices
    if not variants:
        raise ValueError("Empty indices: need at least one variant")
    seq_len = len(variants[0])
    for v in variants:
        if len(v) != seq_len:
            raise ValueError("All semantic variants must share identical length; got mismatch")
    return variants


def build_semantic_prefix_allowed_tokens_fn(
    tokenizer,
    indices: IndicesType,
    sep_text: str = "<|im_start|>assistant\n",
    allow_fallback_full_vocab: bool = True,
    force_eos_after: bool = True,
):
    """Build a prefix_allowed_tokens_fn enforcing per-position token sets.

    Args:
        tokenizer: HF tokenizer.
        indices: semantic variants (list-of-lists or dict values).
        sep_text: marker after which constraints begin.
        allow_fallback_full_vocab: if True, before marker appears, return full vocab.
        force_eos_after: if True, once所有位置填满，仅允许 eos 结束；否则放开词表。
    Returns:
        Callable for `generate`.
    """
    variants = _normalize_indices(indices)

    allowed_tokens_per_pos: Dict[int, set] = {}
    for variant in variants:
        for pos, token_str in enumerate(variant):
            token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                logger.debug("Skip multi-id token '%s' -> %s", token_str, token_ids)
                continue
            allowed_tokens_per_pos.setdefault(pos, set()).add(token_ids[0])

    if not allowed_tokens_per_pos:
        raise ValueError("No single-token constraints extracted; check indices")

    max_pos = max(allowed_tokens_per_pos.keys())
    eos_id = tokenizer.eos_token_id
    vocab_full_cache: List[int] | None = None
    sep_ids = tokenizer(sep_text, add_special_tokens=False).input_ids
    sep_len = len(sep_ids)

    def prefix_allowed_tokens_fn(batch_id: int, sentence):  # sentence: torch.LongTensor
        nonlocal vocab_full_cache
        ids: List[int] = sentence.tolist()

        marker_start = -1
        if sep_len > 0:
            for i in range(len(ids) - sep_len + 1):
                if ids[i:i + sep_len] == sep_ids:
                    marker_start = i
        if marker_start == -1:
            if allow_fallback_full_vocab:
                if vocab_full_cache is None:
                    vocab_full_cache = list(range(tokenizer.vocab_size))
                return vocab_full_cache
            if vocab_full_cache is None:
                vocab_full_cache = list(range(tokenizer.vocab_size))
            return vocab_full_cache

        gen_after_sep = len(ids) - (marker_start + sep_len)

        if gen_after_sep in allowed_tokens_per_pos:
            return list(allowed_tokens_per_pos[gen_after_sep])

        if gen_after_sep > max_pos:
            if force_eos_after and eos_id is not None:
                return [eos_id]
            if vocab_full_cache is None:
                vocab_full_cache = list(range(tokenizer.vocab_size))
            return vocab_full_cache

        if vocab_full_cache is None:
            vocab_full_cache = list(range(tokenizer.vocab_size))
        return vocab_full_cache

    return prefix_allowed_tokens_fn


__all__ = [
    "load_indices_file",
    "build_semantic_prefix_allowed_tokens_fn",
    "build_vllm_semantic_logits_processor",
]


def build_vllm_semantic_logits_processor(
    tokenizer,
    indices: IndicesType,
    sep_text: str = "<|im_start|>assistant\n",
    allow_fallback_full_vocab: bool = True,
    force_eos_after: bool = True,
):
    """Build a vLLM logits processor enforcing per-position allowed token sets.

    This returns a callable with signature

        semantic_position_processor(request_state, prompt_token_ids, output_ids, logits) -> logits

    which can be passed via SamplingParams(logits_processors=[...]). It masks logits so that
    after the separator sequence `sep_text` only tokens belonging to the precomputed allowed
    set of the current semantic position can be sampled.

    Args:
        tokenizer: HF tokenizer used by vLLM (must expose `__call__`, `eos_token_id`, `vocab_size`).
        indices: semantic variants (list-of-lists or dict of lists) identical to HF builder expectations.
        sep_text: textual separator that signals the start of constrained semantic positions.
        allow_fallback_full_vocab: before the separator appears, leave logits unchanged.
        force_eos_after: once all constrained positions are filled, force EOS; otherwise allow full vocab.
    Returns:
        Callable logits processor.
    """
    variants = _normalize_indices(indices)

    allowed_tokens_per_pos: Dict[int, set] = {}
    for variant in variants:
        for pos, token_str in enumerate(variant):
            token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                logger.debug("[vLLM] Skip multi-id token '%s' -> %s", token_str, token_ids)
                continue
            allowed_tokens_per_pos.setdefault(pos, set()).add(token_ids[0])

    if not allowed_tokens_per_pos:
        raise ValueError("No single-token constraints extracted (vLLM); check indices or tokenizer splits")

    max_pos = max(allowed_tokens_per_pos.keys())
    eos_id = tokenizer.eos_token_id
    sep_ids = tokenizer(sep_text, add_special_tokens=False).input_ids
    sep_len = len(sep_ids)

    # Pre-build tensors for faster masking (will be moved to device on first call)
    # Map position -> list of ids (stable order)
    allowed_id_lists: Dict[int, List[int]] = {p: sorted(list(s)) for p, s in allowed_tokens_per_pos.items()}

    def semantic_position_processor(prompt_tokens_ids, past_tokens_ids, logits):
        """vLLM logits processor.

        Args:
            prompt_tokens_ids: List[int] original prompt token ids
            past_tokens_ids: List[int] tokens generated so far (output only)
            logits: torch.Tensor (vocab_size,) or (1, vocab_size)
        Returns:
            Modified logits tensor with disallowed tokens set to -inf when constraint active.
        """
        # Ensure logits is 1D for simpler handling
        reshape_back = False
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits[0]
            reshape_back = True

        combined = list(prompt_tokens_ids) + list(past_tokens_ids)

        # Locate the last occurrence of the separator inside combined
        marker_start = -1
        if sep_len > 0 and len(combined) >= sep_len:
            for i in range(len(combined) - sep_len, -1, -1):  # search backwards for efficiency
                if combined[i:i + sep_len] == sep_ids:
                    marker_start = i
                    break

        if marker_start == -1:
            # Separator not yet seen; optionally leave logits untouched
            # (Could also proactively restrict nothing.)
            return logits.unsqueeze(0) if reshape_back else logits

        gen_after_sep = len(combined) - (marker_start + sep_len)

        # If still within constrained positions
        if gen_after_sep in allowed_id_lists:
            allow_ids = allowed_id_lists[gen_after_sep]
            # Mask everything else to -inf
            mask = torch.full_like(logits, float('-inf'))
            mask[allow_ids] = logits[allow_ids]
            logits = mask
            return logits.unsqueeze(0) if reshape_back else logits

        # Past the last constrained position
        if gen_after_sep > max_pos:
            if force_eos_after and eos_id is not None:
                mask = torch.full_like(logits, float('-inf'))
                mask[eos_id] = logits[eos_id]
                logits = mask
                return logits.unsqueeze(0) if reshape_back else logits
            # else fall back to full vocab
            return logits.unsqueeze(0) if reshape_back else logits

        # Position not in allowed map (gap); fallback behavior
        if not allow_fallback_full_vocab:
            # If strict mode desired, could also zero out all logits (degenerate)
            pass
        return logits.unsqueeze(0) if reshape_back else logits

    return semantic_position_processor

