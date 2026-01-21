"""ML decoder models."""

from __future__ import annotations

from ..registry import register_decoder
from .dummy_model import DummyNeuralDecoder
from .enhancer import (
    BilateralEnhancer,
    CombinedEnhancer,
    NoOpEnhancer,
    SharpenEnhancer,
)
from .residual_espcn import ResidualESPCNEnhancer

# Register built-in enhancement models
register_decoder("noop", NoOpEnhancer)
register_decoder("bilateral", BilateralEnhancer)
register_decoder("sharpen", SharpenEnhancer)
register_decoder("combined", CombinedEnhancer)

# Register neural network models
register_decoder("dummy-neural", DummyNeuralDecoder)
register_decoder("residual-espcn", ResidualESPCNEnhancer)

