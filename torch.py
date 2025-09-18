"""Lightweight subset of the PyTorch API for unit testing purposes.

When a real PyTorch installation is discoverable on ``sys.path`` the module is
re-exported transparently. Otherwise a NumPy-backed shim provides the minimal
surface required by the Prompt Math test-suite.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent


def _load_real_torch() -> object | None:
    """Attempt to import the real ``torch`` module, skipping this shim's path.

    Returns:
        object | None: Imported module when available, otherwise ``None``.
    """

    for entry in list(sys.path):
        try:
            entry_path = Path(entry).resolve()
        except Exception:  # pragma: no cover - non-critical path sanitisation
            continue
        if entry_path == _THIS_DIR:
            continue
        spec = importlib.util.find_spec("torch", [str(entry_path)])
        if spec is None or spec.loader is None:
            continue
        origin = getattr(spec, "origin", None)
        if origin and Path(origin).resolve() == _THIS_FILE:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


if os.environ.get("PROMPT_MATH_FORCE_STUB", "0") not in {"1", "true", "True"}:
    _real = _load_real_torch()
else:
    _real = None

if _real is not None:  # pragma: no cover - exercised only when real torch exists
    sys.modules[__name__] = _real
else:
    import math
    from typing import Iterable, Sequence, Tuple

    import numpy as np

    __all__ = [
        "Tensor",
        "tensor",
        "ones",
        "zeros",
        "full",
        "full_like",
        "zeros_like",
        "ones_like",
        "where",
        "mean",
        "sqrt",
        "clamp",
        "sum",
        "all",
        "eq",
        "stack",
    ]

    float32 = np.float32
    bool_ = np.bool_

    class Tensor(np.ndarray):
        """NumPy ndarray wrapper mimicking the PyTorch tensor API subset."""

        __array_priority__ = 1000

        def __new__(cls, input_array, dtype=None, copy=True):
            """Create a tensor from ``input_array``.

            Args:
                input_array (ArrayLike): Source data to wrap.
                dtype (np.dtype | None, optional): Desired dtype.
                copy (bool, optional): Whether to copy or view the data.

            Returns:
                Tensor: Wrapped NumPy array view.
            """

            if dtype is None:
                array = np.array(input_array, copy=copy)
                if array.dtype == np.float64:
                    array = array.astype(np.float32)
            else:
                array = np.array(input_array, dtype=dtype, copy=copy)
            return np.asarray(array).view(cls)

        def __array_finalize__(self, obj):  # pragma: no cover - numpy protocol hook
            """NumPy finaliser hook; required but unused."""

            return

        def clone(self) -> "Tensor":
            """Tensor: Return a deep copy of the tensor."""

            return Tensor(np.array(self, copy=True))

        def to(self, dtype=None) -> "Tensor":
            """Return a tensor converted to ``dtype`` when provided."""

            if dtype is None:
                return self
            return Tensor(np.array(self, dtype=dtype))

        def item(self):
            """Extract the scalar Python value from a zero-dimensional tensor."""

            return np.asarray(self).item()

    def tensor(data, dtype=None) -> Tensor:
        """Create a tensor from raw ``data``."""

        return Tensor(data, dtype=dtype)

    def ones(*size, dtype=None) -> Tensor:
        """Return a tensor filled with ones."""

        return Tensor(np.ones(size, dtype=dtype if dtype is not None else np.float32))

    def zeros(*size, dtype=None) -> Tensor:
        """Return a tensor filled with zeros."""

        return Tensor(np.zeros(size, dtype=dtype if dtype is not None else np.float32))

    def full(size: Sequence[int] | Tuple[int, ...], fill_value: float, dtype=None) -> Tensor:
        """Return a tensor filled with ``fill_value``."""

        return Tensor(np.full(size, fill_value, dtype=dtype if dtype is not None else np.float32))

    def full_like(tensor_like: Tensor, fill_value: float) -> Tensor:
        """Return a tensor matching ``tensor_like``'s shape filled with ``fill_value``."""

        return Tensor(np.full_like(np.asarray(tensor_like), fill_value))

    def zeros_like(tensor_like: Tensor) -> Tensor:
        """Return a zero tensor of the same shape as ``tensor_like``."""

        return Tensor(np.zeros_like(np.asarray(tensor_like)))

    def ones_like(tensor_like: Tensor) -> Tensor:
        """Return a ones tensor of the same shape as ``tensor_like``."""

        return Tensor(np.ones_like(np.asarray(tensor_like)))

    def where(condition, x, y) -> Tensor:
        """Element-wise ``np.where`` returning a tensor."""

        return Tensor(np.where(np.asarray(condition, dtype=bool), np.asarray(x), np.asarray(y)))

    def mean(value: Tensor, axis: int | None = None) -> Tensor:
        """Return the arithmetic mean along ``axis``."""

        return Tensor(np.mean(np.asarray(value), axis=axis))

    def sum(value: Tensor, axis: int | None = None) -> Tensor:
        """Return the sum along ``axis``."""

        return Tensor(np.sum(np.asarray(value), axis=axis))

    def sqrt(value) -> Tensor:
        """Return the element-wise square root of ``value``."""

        return Tensor(np.sqrt(np.asarray(value)))

    def clamp(value: Tensor, min: float | None = None, max: float | None = None) -> Tensor:
        """Clamp ``value`` to the range ``[min, max]`` when provided."""

        data = np.asarray(value)
        lower = -math.inf if min is None else min
        upper = math.inf if max is None else max
        return Tensor(np.clip(data, lower, upper))

    def stack(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
        """Stack ``tensors`` along ``axis``."""

        arrays = [np.asarray(tensor) for tensor in tensors]
        return Tensor(np.stack(arrays, axis=axis))

    def eq(a, b) -> Tensor:
        """Return an element-wise equality mask between ``a`` and ``b``."""

        return Tensor(np.equal(np.asarray(a), np.asarray(b)), dtype=np.bool_)

    def all(value: Tensor) -> bool:
        """bool: ``True`` only if every element of ``value`` evaluates to ``True``."""

        return bool(np.all(np.asarray(value)))
