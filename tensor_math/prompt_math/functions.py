"""Extended math helpers for advanced prompt manipulation.

The helpers provide vector blending, statistical reductions, masking, and
normalisation utilities shared by Prompt Math custom nodes.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "VectorOperations",
    "StatisticalOperations",
    "MaskingOperations",
    "NormalizationOperations",
]


class VectorOperations:
    """Primitive vector blending helpers."""

    @staticmethod
    def lerp(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
        """Perform linear interpolation between ``a`` and ``b``.

        Args:
            a (torch.Tensor): Start tensor.
            b (torch.Tensor): End tensor.
            weight (float): Normalised interpolation factor.

        Returns:
            torch.Tensor: Interpolated tensor.
        """

        return a + (b - a) * float(weight)

    @staticmethod
    def add_bias(vector: torch.Tensor, bias: float) -> torch.Tensor:
        """Add a scalar ``bias`` to every element of ``vector``."""

        return vector + float(bias)

    @staticmethod
    def scale(vector: torch.Tensor, factor: float) -> torch.Tensor:
        """Multiply ``vector`` by ``factor`` element-wise."""

        return vector * float(factor)


class StatisticalOperations:
    """Statistical aggregation utilities."""

    @staticmethod
    def weighted_mean(vectors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
        """Compute the weighted average of ``vectors``.

        Args:
            vectors (Sequence[torch.Tensor]): Sequence of tensors to average.
            weights (Sequence[float]): Matching weights for each tensor.

        Returns:
            torch.Tensor: Weighted mean tensor.

        Raises:
            ValueError: When the sequences differ in length or the total weight
                is zero.
        """

        if len(vectors) != len(weights):
            raise ValueError("Vectors and weights must be the same length.")
        total_weight = float(sum(weights))
        if total_weight == 0.0:
            raise ValueError("Sum of weights must be non-zero.")
        accumulator = None
        for vector, weight in zip(vectors, weights):
            contrib = vector * float(weight)
            accumulator = contrib if accumulator is None else accumulator + contrib
        return accumulator / total_weight

    @staticmethod
    def mean(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute the arithmetic mean of ``vectors``."""

        if not vectors:
            raise ValueError("At least one vector is required to compute the mean.")
        return StatisticalOperations.weighted_mean(vectors, [1.0] * len(vectors))


class MaskingOperations:
    """Mask application helpers."""

    @staticmethod
    def apply_mask(vector: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """Replace unmasked elements with ``fill_value`` while keeping masked ones."""

        return torch.where(mask, vector, torch.full_like(vector, float(fill_value)))


class NormalizationOperations:
    """Normalization primitives tailored for attention features."""

    @staticmethod
    def layer_norm(vector: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Normalise ``vector`` to have zero mean and unit variance."""

        mean = torch.mean(vector)
        variance = torch.mean((vector - mean) ** 2)
        return (vector - mean) / torch.sqrt(variance + eps)

    @staticmethod
    def clamp(vector: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
        """Clip ``vector`` to the inclusive range ``[min_value, max_value]``."""

        return torch.clamp(vector, min=float(min_value), max=float(max_value))
