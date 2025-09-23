"""Entry point exposing TensorMath to ComfyUI."""

from __future__ import annotations

from pathlib import Path

from .tensor_math.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = str(Path(__file__).parent / 'web')

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
