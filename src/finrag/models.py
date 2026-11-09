"""MODEL_REGISTRY: a name -> a pinned model preset. JobHack's pattern, same three models.

Adding a model is one line here (and its digest in deploy/models.lock); choosing one
is one environment variable, `FINRAG_MODEL`, or the `model` field of an API request:

    FINRAG_MODEL = llama-3b | qwen3-4b | mistral-24b        (default: llama-3b)

All three are open-weight models served by Ollama on our own GPU: no question, filing
passage or answer leaves the infrastructure, and no per-token bill. A hosted model
(GPT, Claude, Mistral's API) would be one more preset kind behind the same calls.

Cost here is GPU time, not tokens: an answer costs the seconds the model spent on it
times the GPU's hourly price. That is the marginal cost; idle time and cold starts are
not included (the backend scales to zero when idle).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPreset:
    name: str
    ollama_tag: str
    # The digest, not the tag, is what eval reports record and the image build checks.
    digest: str
    label: str  # shown to users: vendor, model, size
    # Sent as Ollama's `think` when not None. See qwen3-4b below for why it is True there.
    think: bool | None = None
    # Context for the one-shot answer: 5 passages of ~300 words + prompt fit in 4k.
    # Always explicit (CLAUDE.md): recent Ollama defaults to a context that exhausts memory.
    num_ctx: int = 4096
    # Output cap: bounds latency and GPU cost per answer. An answer that hits it is cut,
    # and a cut answer is refused at the `model` stage, never shown half-written.
    num_predict: int = 1024


MODEL_REGISTRY: dict[str, ModelPreset] = {
    # The baseline: every number in the README up to Week 1 was measured with it.
    "llama-3b": ModelPreset(
        name="llama-3b",
        ollama_tag="llama3.2:3b",
        digest="a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
        label="Meta Llama 3.2 3B",
    ),
    # Same size class, stronger at tool calling. This build ALWAYS reasons (JobHack L84,
    # and here on 2026-11-09: 6,891 output tokens for one answer, then a refusal). Asked
    # not to think, it writes the reasoning into the answer; asked to think, Ollama
    # returns it in a separate field and the answer is clean. So: think=True, and room
    # for the reasoning in both the context and the output budget.
    "qwen3-4b": ModelPreset(
        name="qwen3-4b",
        ollama_tag="qwen3:4b",
        digest="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
        label="Alibaba Qwen3 4B",
        think=True,
        num_ctx=12288,
        num_predict=8192,
    ),
    # Mistral (Apache 2.0), ~15 GB at 4-bit: needs the 24 GB L4, not a 16 GB T4.
    "mistral-24b": ModelPreset(
        name="mistral-24b",
        ollama_tag="mistral-small3.2:24b",
        digest="5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b",
        label="Mistral Small 3.2 24B",
    ),
    # <- ADD A MODEL AS ONE LINE HERE (and its digest in deploy/models.lock)
}
DEFAULT_MODEL = "llama-3b"

# Modal list prices in USD per GPU-hour, read on 2026-11-09 (modal.com/pricing).
# Re-check before quoting a cost: these are inputs to an estimate, not measurements.
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10}

LOCK_PATH = Path(__file__).resolve().parents[2] / "deploy" / "models.lock"


def get(name: str | None = None) -> ModelPreset:
    """The preset called *name* (default: FINRAG_MODEL, then llama-3b). Unknown fails loudly."""
    name = name or os.environ.get("FINRAG_MODEL") or DEFAULT_MODEL
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; options: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def read_lock(path: Path = LOCK_PATH) -> dict[str, str]:
    """deploy/models.lock as {ollama tag: digest}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tag, digest = line.split()
            out[tag] = digest
    return out


@dataclass(frozen=True)
class Usage:
    """What one answer cost: tokens from Ollama's counters, seconds of model time."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    @classmethod
    def from_ollama(cls, model: str, body: dict) -> Usage:
        # Ollama reports durations in nanoseconds; total_duration includes model loading.
        return cls(model, int(body.get("prompt_eval_count") or 0),
                   int(body.get("eval_count") or 0),
                   (body.get("total_duration") or 0) / 1e9, 1)

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.model, self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens,
                     self.seconds + other.seconds, self.calls + other.calls)

    def cost_usd(self, gpu: str) -> float:
        return self.seconds * GPU_USD_PER_HOUR[gpu] / 3600

    def as_dict(self, gpu: str) -> dict:
        return {"model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_calls": self.calls,
                "llm_seconds": round(self.seconds, 2), "gpu": gpu,
                "est_cost_usd": round(self.cost_usd(gpu), 6)}


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop a closed reasoning block if a model writes one into its answer."""
    return THINK_RE.sub("", text).strip()


def ollama_options(preset: ModelPreset, num_ctx: int | None = None) -> dict:
    """The generation fields every call sends: temperature 0, explicit context and cap."""
    body = {"options": {"temperature": 0, "num_ctx": num_ctx or preset.num_ctx,
                        "num_predict": preset.num_predict}}
    if preset.think is not None:
        body["think"] = preset.think
    return body


def unfinished(text: str, done_reason: str | None) -> str | None:
    """Why an answer cannot be shown (cut at the cap, empty, reasoning never closed)."""
    if done_reason == "length":
        return "the answer was cut at the output limit"
    if not text.strip():
        return "the model returned an empty answer"
    if "<think>" in text:
        return "the model's reasoning never ended"
    return None


def detect_gpu(nvidia_name: str) -> str:
    """'NVIDIA A10G' -> 'A10G': the price table key for the GPU that actually ran."""
    for key in GPU_USD_PER_HOUR:
        if key.lower() in nvidia_name.lower():
            return key
    raise ValueError(f"no price for GPU {nvidia_name!r}; add it to GPU_USD_PER_HOUR")
