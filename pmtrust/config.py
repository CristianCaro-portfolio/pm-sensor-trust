"""Typed access to config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]

    @property
    def area(self) -> dict[str, Any]:
        return self.raw["area"]

    @property
    def reference(self) -> dict[str, Any]:
        return self.raw["reference"]

    @property
    def weather(self) -> dict[str, Any]:
        return self.raw["weather"]

    @property
    def qc(self) -> dict[str, Any]:
        return self.raw["qc"]

    @property
    def correction(self) -> dict[str, Any]:
        return self.raw["correction"]

    @property
    def gold(self) -> dict[str, Any]:
        return self.raw["gold"]

    def window(self, name: str) -> tuple[date, date]:
        start, end = self.raw["windows"][name]
        return date.fromisoformat(start), date.fromisoformat(end)

    @property
    def window_names(self) -> list[str]:
        return list(self.raw["windows"])


def load_settings(path: str | Path | None = None) -> Settings:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return Settings(yaml.safe_load(fh))
