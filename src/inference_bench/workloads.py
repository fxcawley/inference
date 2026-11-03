"""Deterministic exact-length prompt assembly with a supplied tokenizer."""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Protocol

from .config import Config
from .provenance import digest_json

GENERATOR_VERSION = "2"
EXCERPT_MARKER = "__INFERENCE_BENCH_EXCERPT_8C53A7__"


def sampling_seed(seed: int, profile: str, repetition: int, phase: str) -> int:
    """Independent of engine/concurrency; leave room for per-request ordinals."""
    return int(digest_json({
        "seed": seed, "profile": profile, "repetition": repetition, "phase": phase,
    })[:7], 16)


class Tokenizer(Protocol):
    bos_token_id: int | None
    all_special_ids: list[int]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def apply_chat_template(self, messages: list[dict], **kwargs) -> str: ...


def normalize_row(text: str) -> str | None:
    if not isinstance(text, str):
        raise ValueError("Corpus rows must contain text")
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text or re.fullmatch(r"=+.*=+", text):
        return None
    return text


def tokenize_corpus(rows: list[str], tokenizer: Tokenizer) -> tuple[list[int], list[dict]]:
    tokens, offsets = [], []
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    for ordinal, row in enumerate(rows):
        text = normalize_row(row)
        if text is None:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        start = len(tokens)
        tokens.extend(ids)
        offsets.append({"row": ordinal, "start": start, "stop": len(tokens)})
        tokens.extend(separator)
    if not tokens:
        raise ValueError("Corpus contains no usable prose")
    return tokens, offsets


def template_parts(tokenizer: Tokenizer, instruction: str) -> tuple[list[int], list[int]]:
    if EXCERPT_MARKER in instruction:
        raise ValueError("Instruction collides with the excerpt marker")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + "\n\n" + EXCERPT_MARKER}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(rendered, str) or rendered.count(EXCERPT_MARKER) != 1:
        raise ValueError("Template must preserve one unambiguous excerpt insertion point")
    prefix_text, suffix_text = rendered.split(EXCERPT_MARKER)
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    if not prefix or not suffix:
        raise ValueError("Template must retain both role prefix and assistant suffix")
    if tokenizer.bos_token_id is not None and (prefix + suffix).count(tokenizer.bos_token_id) != 1:
        raise ValueError("Template must contain exactly one intended BOS")
    return prefix, suffix


def prepare_prompts(
    config: Config, corpus_tokens: list[int], tokenizer: Tokenizer,
) -> tuple[list[dict], dict]:
    prefix, suffix = template_parts(tokenizer, config.data["workload"]["instruction"])
    special = set(tokenizer.all_special_ids)
    prompts = []
    for profile in config.profile_names:
        shape = config.profiles[profile]
        excerpt_size = shape["input_tokens"] - len(prefix) - len(suffix)
        if excerpt_size < 1 or excerpt_size > len(corpus_tokens):
            raise ValueError(f"{profile}: corpus/template cannot satisfy the input budget")
        for repetition in range(1, config.experiment["repetitions"] + 1):
            maximum = max(config.experiment["concurrency"])
            counts = {
                "measurement": config.request_count(maximum),
                "warmup": config.warmup_count(maximum),
            }
            if repetition == 1:
                counts.update(
                    pilot=config.request_count(maximum, phase="pilot"),
                    pilot_warmup=config.warmup_count(maximum),
                )
            for phase, count in counts.items():
                identity = {
                    "seed": config.experiment["seed"], "profile": profile,
                    "repetition": repetition, "phase": phase,
                    "generator": GENERATOR_VERSION,
                }
                rng = random.Random(int(digest_json(identity)[:16], 16))
                for ordinal in range(count):
                    for _ in range(1000):
                        start = rng.randrange(len(corpus_tokens) - excerpt_size + 1)
                        excerpt = corpus_tokens[start:start + excerpt_size]
                        if not special.intersection(excerpt):
                            break
                    else:
                        raise ValueError("Unable to select a corpus window without role/special tokens")
                    ids = prefix + excerpt + suffix
                    assert len(ids) == shape["input_tokens"]
                    prompts.append({
                        "request_id": f"{phase}:{profile}:r{repetition}:{ordinal:04d}",
                        "profile": profile, "repetition": repetition, "phase": phase,
                        "prompt_ids": ids, "prompt_sha256": digest_json(ids),
                        "input_tokens": len(ids), "source_start": start,
                        "source_stop": start + excerpt_size,
                    })
    manifest = {
        "schema_version": 1, "generator_version": GENERATOR_VERSION,
        "config_sha256": config.fingerprint, "corpus_tokens_sha256": digest_json(corpus_tokens),
        "prefix_ids": prefix, "suffix_ids": suffix,
        "prefix_sha256": digest_json(prefix), "suffix_sha256": digest_json(suffix),
        "normalization": "NFC; collapse whitespace; omit empty and heading-only rows",
        "prompts_sha256": digest_json(prompts), "count": len(prompts),
    }
    return prompts, manifest


def select_prompts(
    prompts: list[dict], profile: str, repetition: int, phase: str, count: int,
) -> list[dict]:
    selected = [
        p for p in prompts if p["profile"] == profile and p["repetition"] == repetition
        and p["phase"] == phase
    ][:count]
    if len(selected) != count:
        raise ValueError("Prepared prompt set cannot supply the requested cell")
    return selected
