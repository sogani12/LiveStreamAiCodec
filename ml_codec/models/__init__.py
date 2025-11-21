"""ML decoder models."""

from __future__ import annotations

from ..registry import register_decoder
from .enhancer import (
    BilateralEnhancer,
    CombinedEnhancer,
    NoOpEnhancer,
    SharpenEnhancer,
)

# Register built-in enhancement models
register_decoder("noop", NoOpEnhancer)
register_decoder("bilateral", BilateralEnhancer)
register_decoder("sharpen", SharpenEnhancer)
register_decoder("combined", CombinedEnhancer)

