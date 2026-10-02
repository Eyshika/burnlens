"""Configurable waste algorithm.

The waste algorithm is configuration, not code. Every
threshold, the set of enabled rules, which models count as premium, and the task-to-tier
table live in one TOML file. Nothing in the rules is hard-coded to a vendor.

Lookup order: --config, $BURNLENS_CONFIG, ./burnlens.toml, ~/.burnlens/burnlens.toml.

    [thresholds]
    context_tokens = 150000
    large_payload_bytes = 50000
    live_burn_warn_per_min = 1000000

    [rules]
    disabled = ["long-session"]

    [models]
    premium_markers = ["opus", "fable", "gpt-5"]

    [tiers]
    lookup = "haiku"
    research = "sonnet"
    debug = "opus"

    [prices]
    cache_write = 1.25
    cache_read = 0.1
    output = 5.0

    [prices.fable]          # matched against the model id; cache reads are 0.025x the base here
    cache_read = 0.025
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .findings import Thresholds
from .model import Usage

logger = logging.getLogger(__name__)

DEFAULT_LOCATIONS: tuple[Path, ...] = (Path("burnlens.toml"), Path.home() / ".burnlens" / "burnlens.toml")


class ConfigError(ValueError):
    """Raised for a config file that exists but cannot be applied."""


@dataclass(frozen=True)
class TokenPrices:
    """Relative cost of each token class, fresh input = 1.0.

    Defaults are the published API ratios. Subscription quota accounting is not public, so
    these weigh one class against another and are never a bill.
    """

    input: float = 1.0
    cache_write: float = 1.25
    cache_read: float = 0.1
    output: float = 5.0

    def weigh(self, usage: Usage) -> dict[str, float]:
        return {
            "input": usage.input_tokens * self.input,
            "cache_write": usage.cache_creation_input_tokens * self.cache_write,
            "cache_read": usage.cache_read_input_tokens * self.cache_read,
            "output": usage.output_tokens * self.output,
        }

    def as_dict(self) -> dict[str, float]:
        return {"input": self.input, "cache_write": self.cache_write, "cache_read": self.cache_read, "output": self.output}


PRICES_FILE = Path(__file__).with_name("prices.toml")


@dataclass(frozen=True)
class PriceTable:
    """Per-model weights, derived from published prices in prices.toml.

    Cache-read is not a fixed fraction of base input across models, so a fleet running several
    models has to weigh each model's tokens with that model's own ratios. The table carries the
    date it was checked, because a price nobody has re-checked is wrong rather than approximate.
    """

    default: TokenPrices = field(default_factory=TokenPrices)
    by_marker: dict[str, TokenPrices] = field(default_factory=dict)
    overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    as_of: str = ""
    stale_after_days: int = 90

    @property
    def stale_days(self) -> int | None:
        """Days past the staleness window, or None when the table is current or undated."""
        if not self.as_of:
            return None
        try:
            checked = date.fromisoformat(self.as_of)
        except ValueError:
            return None
        overdue = (date.today() - checked).days - self.stale_after_days
        return overdue if overdue > 0 else None

    def provenance(self) -> str:
        if not self.as_of:
            return "prices undated; weighted figures have no basis"
        stale = self.stale_days
        return f"prices checked {self.as_of}" + (f", {stale} days past due for a re-check" if stale else "")

    def for_model(self, model: str) -> TokenPrices:
        """Published entry for the closest-matching model, then the user's override on top."""
        lowered = (model or "").lower()
        published = _longest_match(self.by_marker, lowered)
        prices = self.by_marker[published] if published else self.default
        override = _longest_match(self.overrides, lowered)
        return dataclasses.replace(prices, **self.overrides[override]) if override else prices

    def weigh(self, by_model: dict[str, Usage]) -> dict[str, float]:
        totals = {"input": 0.0, "cache_write": 0.0, "cache_read": 0.0, "output": 0.0}
        for model, usage in by_model.items():
            for name, value in self.for_model(model).weigh(usage).items():
                totals[name] += value
        return totals

    def as_dict(self) -> dict[str, object]:
        return {
            "default": self.default.as_dict(),
            "by_model": {m: p.as_dict() for m, p in self.by_marker.items()},
            "overrides": {m: dict(v) for m, v in self.overrides.items()},
            "as_of": self.as_of,
            "stale_days": self.stale_days,
            "provenance": self.provenance(),
        }


@dataclass(frozen=True)
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    disabled_rules: frozenset[str] = frozenset()
    task_tiers: dict[str, str] = field(default_factory=dict)
    prices: PriceTable = field(default_factory=PriceTable)
    source: Path | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "thresholds": dataclasses.asdict(self.thresholds),
            "disabled_rules": sorted(self.disabled_rules),
            "task_tiers": dict(self.task_tiers),
            "prices": self.prices.as_dict(),
            "source": str(self.source) if self.source else None,
        }


def find_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    env = os.environ.get("BURNLENS_CONFIG")
    if env:
        path = Path(env)
        if not path.is_file():
            raise ConfigError(f"BURNLENS_CONFIG points to a missing file: {path}")
        return path
    for candidate in DEFAULT_LOCATIONS:
        if candidate.is_file():
            return candidate
    return None


def load_settings(explicit: Path | None = None, overrides: dict[str, object] | None = None) -> Settings:
    """Defaults, then the config file, then explicit overrides (CLI flags)."""
    path = find_config(explicit)
    raw: dict[str, object] = {}
    if path is not None:
        try:
            raw = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.error("cannot read config %s: %s", path, exc)
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
    th_values: dict[str, object] = {}
    known = {f.name for f in dataclasses.fields(Thresholds)}
    for key, value in (raw.get("thresholds") or {}).items():
        if key not in known:
            raise ConfigError(f"unknown threshold {key!r} in {path}; known: {sorted(known)}")
        th_values[key] = value
    models = raw.get("models") or {}
    if "premium_markers" in models:
        th_values["premium_markers"] = tuple(str(m).lower() for m in models["premium_markers"])
    for key, value in (overrides or {}).items():
        if value is not None:
            th_values[key] = value
    rules = raw.get("rules") or {}
    disabled = frozenset(str(r) for r in rules.get("disabled", []))
    tiers = {str(k): str(v) for k, v in (raw.get("tiers") or {}).items()}
    return Settings(
        thresholds=Thresholds(**th_values),
        disabled_rules=disabled,
        task_tiers=tiers,
        prices=_price_table(raw.get("prices") or {}, path),
        source=path,
    )


def load_published_prices(path: Path = PRICES_FILE) -> PriceTable:
    """Published USD-per-million prices, normalised to weights against each model's own base input."""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.error("cannot read %s: %s", path, exc)
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    by_marker: dict[str, TokenPrices] = {}
    for model, prices in (raw.get("models") or {}).items():
        base = float(prices.get("input") or 0.0)
        if base <= 0:
            raise ConfigError(f"{path}: model {model!r} has no input price to normalise against")
        by_marker[str(model).lower()] = TokenPrices(
            input=1.0,
            cache_write=float(prices.get("cache_write") or 0.0) / base,
            cache_read=float(prices.get("cache_read") or 0.0) / base,
            output=float(prices.get("output") or 0.0) / base,
        )
    return PriceTable(
        by_marker=by_marker,
        as_of=str(raw.get("as_of") or ""),
        stale_after_days=int(raw.get("stale_after_days") or 90),
    )


def _price_table(raw: dict[str, object], path: Path | None) -> PriceTable:
    """Published prices are the base; a [prices] section in the user's config overrides them."""
    published = load_published_prices()
    known = {f.name for f in dataclasses.fields(TokenPrices)}
    base: dict[str, float] = {}
    markers: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            markers[str(key).lower()] = {str(k): float(v) for k, v in value.items()}
            continue
        if key not in known:
            raise ConfigError(f"unknown price {key!r} in {path}; known: {sorted(known)}")
        base[str(key)] = float(value)
    for marker, values in markers.items():
        unknown = set(values) - known
        if unknown:
            raise ConfigError(f"unknown price {sorted(unknown)} for model {marker!r} in {path}; known: {sorted(known)}")
    return PriceTable(
        default=dataclasses.replace(published.default, **base) if base else published.default,
        by_marker=dict(published.by_marker),
        overrides=markers,
        as_of=published.as_of,
        stale_after_days=published.stale_after_days,
    )


def _longest_match(markers: dict, model: str) -> str | None:
    """Most specific marker wins, so claude-fable-5 beats fable on claude-fable-5-1."""
    hits = [m for m in markers if m in model]
    return max(hits, key=len) if hits else None
