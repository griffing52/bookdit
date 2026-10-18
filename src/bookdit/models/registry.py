"""Name -> model class, resolved lazily inside :func:`build_model`.

The release ships one backbone: the flow-matching diffusion transformer in `flow.py`.
"""
from __future__ import annotations

from torch import nn

# name -> attribute in .flow. "wandit" is the name stored in the `arch` key of the paper's
# checkpoint, so it must keep resolving; "bookdit" is the same class under the paper's name.
_FLOW: dict[str, str] = {
    "bookdit": "BookDiT",
    "wandit": "BookDiT",
}


def available_models() -> list[str]:
    """Sorted list of every name `build_model` accepts."""
    return sorted(_FLOW)


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a backbone by name. Raises KeyError naming the valid options."""
    key = name.lower()
    if key not in _FLOW:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}")
    from . import flow
    return getattr(flow, _FLOW[key])(**kwargs)
