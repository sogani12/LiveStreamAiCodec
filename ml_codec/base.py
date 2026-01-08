"""
Base classes and utilities for ML codec components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseMLDecoder(ABC):
    """
    Abstract base class for decoder-side ML enhancement modules.

    Subclasses should implement `enhance`, and optionally override
    `setup` / `teardown` for resource management (models, GPU contexts, etc.).
    """

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._is_setup = False

    def setup(self) -> None:
        """
        Prepare resources (load weights, init runtime). Called lazily before first use.
        """
        self._is_setup = True

    def teardown(self) -> None:
        """
        Release resources (close sessions, free GPU memory).
        """
        self._is_setup = False

    def ensure_setup(self) -> None:
        if not self._is_setup:
            self.setup()

    @abstractmethod
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Run enhancement on a single BGR frame and return a new frame.
        """
        raise NotImplementedError

    def enhance_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Optional batch helper, defaults to per-frame enhancement.
        """
        return [self.enhance(frame) for frame in frames]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseMLDecoder":
        return cls(**config)



