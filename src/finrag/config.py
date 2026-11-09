"""Settings, read from the environment (or a .env file at the project root).

Kept as a plain dataclass: a handful of values does not need a settings framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # SEC asks every automated client to identify itself with a name and a contact
    # address (https://www.sec.gov/os/accessing-edgar-data). No default on purpose.
    sec_user_agent: str = field(default_factory=lambda: os.environ.get("SEC_USER_AGENT", ""))
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FINRAG_DATA", ROOT / "data"))
    )
    collection: str = os.environ.get("FINRAG_COLLECTION", "sec_filings")
    embed_model: str = os.environ.get("FINRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("FINRAG_LLM", "llama3.2:3b")
    top_k: int = int(os.environ.get("FINRAG_TOP_K", "5"))
    # Below this cosine score the best passage is not treated as evidence.
    # 0.72: calibrated on eval/suites/seed.jsonl, real 10-Ks (2026-11-07). Lowest
    # answerable best score 0.748, highest separable unanswerable 0.700. Tuned on the
    # seed suite, so NOT held-out; re-check on the Week 2 suite. See README "Status".
    min_score: float = float(os.environ.get("FINRAG_MIN_SCORE", "0.72"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def sec_names_path(self) -> Path:
        return self.data_dir / "company_tickers.json"


def get_settings() -> Settings:
    return Settings()
