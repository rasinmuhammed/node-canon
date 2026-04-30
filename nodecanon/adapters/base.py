from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from nodecanon.core.models import KGGraph


class BaseAdapter(ABC):
    """Convert between a framework's native graph format and KGGraph."""

    @abstractmethod
    def load(self, source: Path | str | Any) -> KGGraph:
        raise NotImplementedError

    @abstractmethod
    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        raise NotImplementedError
