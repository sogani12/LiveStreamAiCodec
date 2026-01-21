"""
Simple registry for ML decoder models.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import BaseMLDecoder


ModelFactory = Callable[..., BaseMLDecoder]

_DECODER_REGISTRY: Dict[str, ModelFactory] = {}


def register_decoder(name: str, factory: ModelFactory) -> None:
    """
    Register a decoder model factory under a given name.
    """
    key = name.lower()
    if key in _DECODER_REGISTRY:
        raise ValueError(f"Decoder '{name}' already registered")
    _DECODER_REGISTRY[key] = factory


def get_decoder(name: str, **config) -> BaseMLDecoder:
    key = name.lower()
    if key not in _DECODER_REGISTRY:
        raise KeyError(f"Decoder '{name}' not found. Available: {list(_DECODER_REGISTRY.keys())}")
    return _DECODER_REGISTRY[key](**config)


def list_decoders() -> list[str]:
    return sorted(_DECODER_REGISTRY.keys())




