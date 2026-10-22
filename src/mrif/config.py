"""Typed configuration loaded from TOML. The packaged ``default.toml`` holds the defaults."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    source: str = "yahoo"
    start: str = "2007-04-11"
    end: str = "2025-12-31"
    cache_dir: str = "data"


@dataclass
class RiskConfig:
    window: int = 252
    var_levels: list[float] = field(default_factory=lambda: [0.99, 0.975])
    es_level: float = 0.975
    ewma_lambda: float = 0.94


@dataclass
class FeatureConfig:
    vol_window: int = 21
    corr_window: int = 63
    drawdown_window: int = 252
    duration_pair: list[str] = field(default_factory=lambda: ["equities", "rates"])
    risk_assets: list[str] = field(default_factory=lambda: ["equities", "credit", "commodities"])


@dataclass
class RegimeConfig:
    episodes: list[list[str]] = field(default_factory=list)


@dataclass
class ClassifierConfig:
    C: float = 1.0
    class_weight: str | None = "balanced"
    first_oos_year: int = 2012
    threshold: float = 0.5


@dataclass
class GovernanceConfig:
    runs_dir: str = "runs"


@dataclass
class Config:
    data: DataConfig
    assets: dict[str, str]
    weights: dict[str, float]
    risk: RiskConfig
    features: FeatureConfig
    regime: RegimeConfig
    classifier: ClassifierConfig
    governance: GovernanceConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls(
            data=DataConfig(**raw.get("data", {})),
            assets=dict(raw["assets"]),
            weights={k: float(v) for k, v in raw["weights"].items()},
            risk=RiskConfig(**raw.get("risk", {})),
            features=FeatureConfig(**raw.get("features", {})),
            regime=RegimeConfig(**raw.get("regime", {})),
            classifier=ClassifierConfig(**raw.get("classifier", {})),
            governance=GovernanceConfig(**raw.get("governance", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    @classmethod
    def default(cls) -> "Config":
        text = resources.files("mrif").joinpath("default.toml").read_text()
        return cls.from_dict(tomllib.loads(text))

    @property
    def asset_classes(self) -> list[str]:
        return list(self.assets)

    def weight_vector(self) -> list[float]:
        return [self.weights[a] for a in self.asset_classes]

    def validate(self) -> None:
        if set(self.assets) != set(self.weights):
            raise ValueError("[assets] and [weights] must list the same asset classes")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"portfolio weights must sum to 1, got {total:.6f}")
        for name in ("duration_pair", "risk_assets"):
            missing = [a for a in getattr(self.features, name) if a not in self.assets]
            if missing:
                raise ValueError(f"features.{name} refers to unknown asset classes: {missing}")
        if self.risk.window < 30:
            raise ValueError("risk.window is too short for stable covariance estimates")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))
