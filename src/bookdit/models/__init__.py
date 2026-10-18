"""BookDiT model backbones.

The flow-matching diffusion transformer lives in `.flow`; `build_model` resolves it by name.
"""
from .registry import available_models, build_model

__all__ = ["available_models", "build_model"]
