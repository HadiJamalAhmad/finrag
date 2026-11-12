"""MODEL_REGISTRY: a name -> a model preset. JobHack's pattern, plus two hosted models.

Adding a model is one line here (an Ollama model also needs its digest in
deploy/models.lock); choosing one is one environment variable, `FINRAG_MODEL`, the
`--model` flag, or the `model` field of an API request:

    FINRAG_MODEL = llama-3b | qwen3-4b | mistral-24b | gpt-luna | claude-haiku

Two kinds of preset behind the same calls (answer.generate, the agent's chat model):
- provider "ollama": open weights on our own GPU, pinned by digest. No question or
  filing passage leaves the infrastructure. Cost = model seconds x the GPU's price.
- provider "openai" / "anthropic": hosted. The question and the retrieved passages
  (public SEC text) go to the vendor, so these are opt-in and need the vendor's key.
  A dated model ID is the provenance (there is no digest). Cost = tokens x list price.
Both costs are marginal: idle time and cold starts are not included.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_id: str  # Ollama tag, or the vendor's model ID
    # Ollama: the digest, not the tag, is what eval reports record and the image build
    # checks. Hosted models have none (None).
    digest: str | None
    label: str  # shown to users: vendor, model, size
    provider: str = "ollama"  # ollama | openai | anthropic
    # Hosted models: list price in USD per million (input, output) tokens.
    usd_per_mtok: tuple[float, float] | None = None
    # Hosted models: extra request fields, e.g. (("reasoning_effort", "none"),).
    api_params: tuple[tuple[str, object], ...] = ()
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
        model_id="llama3.2:3b",
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
        model_id="qwen3:4b",
        digest="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
        label="Alibaba Qwen3 4B",
        think=True,
        num_ctx=12288,
        num_predict=8192,
    ),
    # Mistral (Apache 2.0), ~15 GB at 4-bit: needs the 24 GB L4, not a 16 GB T4.
    "mistral-24b": ModelPreset(
        name="mistral-24b",
        model_id="mistral-small3.2:24b",
        digest="5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b",
        label="Mistral Small 3.2 24B",
    ),
    # Hosted, opt-in. Prices and IDs read on 2026-11-09 from the vendors' model pages
    # (openai.com/api/pricing answered 403: OpenAI prices are from its model page).
    # Neither gets `temperature`: Claude 5.5 rejects it ("omit temperature"), and the
    # GPT-6 pages do not say. Re-check prices before quoting a cost.
    # OpenAI's cost-efficient model. Function calling in Chat Completions requires
    # reasoning_effort "none" (model page), which also keeps latency down.
    "gpt-luna": ModelPreset(
        name="gpt-luna",
        model_id="gpt-6-luna",
        digest=None,
        label="OpenAI GPT-6 Luna",
        provider="openai",
        usd_per_mtok=(0.10, 0.50),
        api_params=(("reasoning_effort", "none"),),
        num_predict=4096,
    ),
    # Anthropic's fastest model, same list price as Luna (prompts under 100K tokens).
    # Adaptive thinking may spend output tokens before the text: a larger cap.
    "claude-haiku": ModelPreset(
        name="claude-haiku",
        model_id="claude-haiku-5-5",
        digest=None,
        label="Anthropic Claude Haiku 5.5",
        provider="anthropic",
        usd_per_mtok=(0.10, 0.50),
        num_predict=4096,
    ),
    # <- ADD A MODEL AS ONE LINE HERE (Ollama: and its digest in deploy/models.lock)
}
# The presets on our own GPU (the image pulls them; the bake-off's default set).
OLLAMA_MODELS = [n for n, p in MODEL_REGISTRY.items() if p.provider == "ollama"]
DEFAULT_MODEL = "llama-3b"

# Modal list prices in USD per GPU-hour, read on 2026-11-09 (modal.com/pricing).
# Re-check before quoting a cost: these are inputs to an estimate, not measurements.
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10,
                    # Oracle Cloud Always Free, Ampere A1 (CPU, no GPU): no marginal cost.
                    # The free fallback while Modal is off (deploy/oracle/).
                    "A1-CPU": 0.0}

LOCK_PATH = Path(__file__).resolve().parents[2] / "deploy" / "models.lock"


def get(name: str | None = None) -> ModelPreset:
    """The preset called *name* (default: FINRAG_MODEL, then llama-3b). Unknown fails loudly."""
    name = name or os.environ.get("FINRAG_MODEL") or DEFAULT_MODEL
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; options: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def read_lock(path: Path = LOCK_PATH) -> dict[str, str]:
    """deploy/models.lock as {Ollama tag: digest}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tag, digest = line.split()
            out[tag] = digest
    return out


@dataclass(frozen=True)
class Usage:
    """What one answer cost: tokens from the provider's counters, seconds of model time."""

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
        """Hosted: tokens x list price. Ollama: model seconds x the GPU's hourly price."""
        price = MODEL_REGISTRY[self.model].usd_per_mtok if self.model in MODEL_REGISTRY else None
        if price:
            return (self.prompt_tokens * price[0] + self.completion_tokens * price[1]) / 1e6
        return self.seconds * GPU_USD_PER_HOUR[gpu] / 3600

    def as_dict(self, gpu: str) -> dict:
        hosted = self.model in MODEL_REGISTRY and MODEL_REGISTRY[self.model].usd_per_mtok
        return {"model": self.model, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_calls": self.calls,
                "llm_seconds": round(self.seconds, 2), "gpu": None if hosted else gpu,
                "pricing": "tokens" if hosted else "gpu-time",
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
    """'NVIDIA L4' -> 'L4': the price table key for the GPU that actually ran.

    Modal's A10G reports itself to nvidia-smi as "NVIDIA A10" (seen 2026-11-09).
    """
    if nvidia_name.strip().upper().endswith(" A10"):
        return "A10G"
    for key in GPU_USD_PER_HOUR:
        if key.lower() in nvidia_name.lower():
            return key
    raise ValueError(f"no price for GPU {nvidia_name!r}; add it to GPU_USD_PER_HOUR")


def model_parameters(preset: ModelPreset) -> dict:
    """What the generation was asked with, recorded on its Langfuse observation."""
    if preset.provider == "ollama":
        params = {"temperature": 0, "num_ctx": preset.num_ctx, "num_predict": preset.num_predict}
        if preset.think is not None:
            params["think"] = preset.think
        return params
    return {"max_output_tokens": preset.num_predict,
            **{k: v for k, v in preset.api_params if isinstance(v, str | int | float | bool)}}
