"""Scheduling primitives used by the Prompt Math custom node.

The module provides easing curves, schedule containers, and helpers that map
prompt tokens onto time-based weights compatible with ComfyUI attention flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Sequence

__all__ = [
    "CurveType",
    "ScheduleParams",
    "TokenSchedule",
    "CurveFunctions",
    "TimestepConverter",
    "ScheduleEvaluator",
    "AttentionScheduler",
    "create_fade_in_schedule",
    "create_fade_out_schedule",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` between ``low`` and ``high``.

    Args:
        value (float): Candidate value to bound.
        low (float, optional): Minimum allowed value. Defaults to ``0.0``.
        high (float, optional): Maximum allowed value. Defaults to ``1.0``.

    Returns:
        float: Bounded value constrained to the inclusive ``[low, high]`` range.
    """

    return max(low, min(high, value))


class CurveType(str, Enum):
    """Enumerates easing curves supported by the scheduler."""

    LINEAR = "linear"
    SMOOTH = "smooth"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    CONSTANT = "constant"


class CurveFunctions:
    """Collection of easing functions used to shape schedule weights."""

    @staticmethod
    def linear(t: float, start: float = 0.0, end: float = 1.0) -> float:
        """Interpolate linearly between ``start`` and ``end``.

        Args:
            t (float): Normalised time in ``[0, 1]``.
            start (float, optional): Weight at ``t = 0``. Defaults to ``0.0``.
            end (float, optional): Weight at ``t = 1``. Defaults to ``1.0``.

        Returns:
            float: Interpolated weight for ``t``.
        """

        return start + (end - start) * t

    @staticmethod
    def smooth(t: float, start: float = 0.0, end: float = 1.0) -> float:
        """Smoothly ease between ``start`` and ``end`` using Hermite blending."""

        hermite = t * t * (3.0 - 2.0 * t)
        return start + (end - start) * hermite

    @staticmethod
    def ease_in(t: float, start: float = 0.0, end: float = 1.0) -> float:
        """Bias interpolation toward the end of the interval."""

        eased = t * t
        return start + (end - start) * eased

    @staticmethod
    def ease_out(t: float, start: float = 0.0, end: float = 1.0) -> float:
        """Bias interpolation toward the start of the interval."""

        eased = 1.0 - (1.0 - t) * (1.0 - t)
        return start + (end - start) * eased

    @staticmethod
    def ease_in_out(t: float, start: float = 0.0, end: float = 1.0) -> float:
        """Blend between ease-in and ease-out segments."""

        if t < 0.5:
            eased = 2.0 * t * t
        else:
            eased = 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
        return start + (end - start) * eased

    @staticmethod
    def constant(_: float, start: float = 0.0, __: float = 1.0) -> float:
        """Return a constant ``start`` weight regardless of time."""

        return start

    _CURVE_MAP = {
        CurveType.LINEAR: linear.__func__,
        CurveType.SMOOTH: smooth.__func__,
        CurveType.EASE_IN: ease_in.__func__,
        CurveType.EASE_OUT: ease_out.__func__,
        CurveType.EASE_IN_OUT: ease_in_out.__func__,
        CurveType.CONSTANT: constant.__func__,
    }

    @classmethod
    def resolve(cls, curve_type: CurveType) -> Callable[[float, float, float], float]:
        """Resolve a :class:`CurveType` into its easing function.

        Args:
            curve_type (CurveType): Curve identifier to resolve.

        Returns:
            Callable[[float, float, float], float]: Matching easing implementation.

        Raises:
            ValueError: If the curve type is unsupported.
        """

        try:
            return cls._CURVE_MAP[curve_type]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported curve type: {curve_type}") from exc


def _normalize_curve(curve: CurveType | str | None) -> CurveType:
    """Normalise user-provided curve tokens to :class:`CurveType` members.

    Args:
        curve (CurveType | str | None): Raw curve input originating from
            parsing or user configuration.

    Returns:
        CurveType: Concrete curve enumeration.

    Raises:
        ValueError: If the provided curve cannot be resolved.
    """

    if curve is None:
        return CurveType.LINEAR
    if isinstance(curve, CurveType):
        return curve
    normalized = str(curve).strip().lower()
    for candidate in CurveType:
        if candidate.value == normalized or candidate.name.lower() == normalized:
            return candidate
    raise ValueError(f"Unknown curve type: {curve}")


@dataclass(frozen=True)
class ScheduleParams:
    """Configuration parameters shared by token schedules.

    Attributes:
        start_time (float): Normalised start of the transition.
        end_time (float): Normalised end of the transition.
        curve_type (CurveType): Easing curve to apply between start and end.
        clamp_output (bool): Whether to clamp computed weights into ``[0, 1]``.
    """

    start_time: float
    end_time: float
    curve_type: CurveType = CurveType.LINEAR
    clamp_output: bool = True

    def __post_init__(self) -> None:
        """Validate that the schedule covers a positive duration."""

        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time.")


@dataclass
class TokenSchedule:
    """Associates a token with timing metadata used for attention blending.

    Attributes:
        token_expr (str): Token expression or literal identifier.
        token_indices (Sequence[int] | Sequence[str]): Token indices resolved by
            the embedding backend.
        params (ScheduleParams): Timing configuration for the schedule.
        direction (str): Indicates ``"fade_in"`` or ``"fade_out"`` behaviour.
        metadata (dict[str, object]): Additional application-specific metadata.
    """

    token_expr: str
    token_indices: Sequence[int] | Sequence[str]
    params: ScheduleParams
    direction: str = "fade_in"
    metadata: dict[str, object] = field(default_factory=dict)

    def weight_at(self, time: float) -> float:
        """Compute the schedule weight at the supplied time point.

        Args:
            time (float): Normalised time in the inclusive range ``[0, 1]``.

        Returns:
            float: Weight respecting the configured direction and clamping.
        """

        start, end = self.params.start_time, self.params.end_time
        if time <= start:
            base = 0.0
        elif time >= end:
            base = 1.0
        else:
            span = end - start
            normalized = (time - start) / span
            normalized = _clamp(normalized)
            curve = CurveFunctions.resolve(self.params.curve_type)
            base = curve(normalized, 0.0, 1.0)
        weight = base if self.direction == "fade_in" else 1.0 - base
        if self.params.clamp_output:
            weight = _clamp(weight)
        return weight


class TimestepConverter:
    """Utility for translating between raw scheduler units and percentages."""

    @staticmethod
    def to_normalized(
        value: float,
        mode: str,
        *,
        total_steps: int | None = None,
        max_time: float | None = None,
    ) -> float:
        """Convert a raw scheduler value into a normalised percentage.

        Args:
            value (float): Source value such as a diffusion step count.
            mode (str): Conversion mode ``"step_based"``, ``"time_based"``, or
                ``"percent"``.
            total_steps (int | None, optional): Total number of steps when using
                ``"step_based"`` mode.
            max_time (float | None, optional): Maximum time horizon when using
                ``"time_based"`` mode.

        Returns:
            float: Value normalised to ``[0, 1]``.

        Raises:
            ValueError: If the mode is unsupported or required metadata is
                missing.
        """

        if mode == "step_based":
            if not total_steps or total_steps <= 0:
                raise ValueError("total_steps must be a positive integer for step_based mode.")
            return value / float(total_steps)
        if mode == "time_based":
            if not max_time or max_time <= 0:
                raise ValueError("max_time must be a positive number for time_based mode.")
            return value / float(max_time)
        if mode == "percent":
            return value
        raise ValueError(f"Unsupported timestep mode: {mode}")

    @staticmethod
    def from_normalized(
        value: float,
        mode: str,
        *,
        total_steps: int | None = None,
        max_time: float | None = None,
    ) -> float:
        """Convert a normalised percentage back into scheduler units.

        Args:
            value (float): Normalised value in ``[0, 1]``.
            mode (str): Conversion mode ``"step_based"``, ``"time_based"`` or
                ``"percent"``.
            total_steps (int | None, optional): Step count for ``"step_based"``
                mode.
            max_time (float | None, optional): Time horizon for ``"time_based"``
                mode.

        Returns:
            float: Value rescaled into scheduler-native units.

        Raises:
            ValueError: If the mode is unsupported or required metadata is
                missing.
        """

        if mode == "step_based":
            if not total_steps or total_steps <= 0:
                raise ValueError("total_steps must be provided for step_based mode.")
            return value * float(total_steps)
        if mode == "time_based":
            if not max_time or max_time <= 0:
                raise ValueError("max_time must be provided for time_based mode.")
            return value * float(max_time)
        if mode == "percent":
            return value
        raise ValueError(f"Unsupported timestep mode: {mode}")


class ScheduleEvaluator:
    """Evaluate a schedule at a point in time."""

    def evaluate(self, schedule: TokenSchedule, time: float) -> float:
        """Return the weight produced by ``schedule`` at ``time``."""

        return schedule.weight_at(time)


class AttentionScheduler:
    """Aggregates schedules and exposes helper queries for attention flows."""

    def __init__(self) -> None:
        """Initialise the scheduler with an empty registry."""

        self._schedules: List[TokenSchedule] = []

    def register(self, schedule: TokenSchedule) -> TokenSchedule:
        """Store a schedule and return it for chaining.

        Args:
            schedule (TokenSchedule): Schedule instance to register.

        Returns:
            TokenSchedule: The registered schedule, enabling fluent usage.
        """

        self._schedules.append(schedule)
        return schedule

    def clear(self) -> None:
        """Remove all registered schedules."""

        self._schedules.clear()

    def schedules_for(self, token_expr: str | None = None) -> List[TokenSchedule]:
        """Retrieve schedules scoped to a token expression.

        Args:
            token_expr (str | None, optional): If provided, only schedules
                matching the expression are returned. When ``None``, every
                schedule is returned.

        Returns:
            List[TokenSchedule]: Matching schedules.
        """

        if token_expr is None:
            return list(self._schedules)
        return [schedule for schedule in self._schedules if schedule.token_expr == token_expr]

    def weight_at(self, token_expr: str, time: float) -> float:
        """Compute the strongest weight for ``token_expr`` at ``time``.

        Args:
            token_expr (str): Token expression to query.
            time (float): Normalised time position.

        Returns:
            float: Maximum weight across registered schedules, defaulting to
            ``1.0`` when none are found.
        """

        schedules = self.schedules_for(token_expr)
        if not schedules:
            return 1.0
        evaluator = ScheduleEvaluator()
        return max(evaluator.evaluate(schedule, time) for schedule in schedules)


def create_fade_in_schedule(
    token_expr: str,
    token_indices: Sequence[int] | Sequence[str],
    start_time: float,
    end_time: float,
    curve_type: CurveType | str | None = None,
    **metadata: object,
) -> TokenSchedule:
    """Convenience factory for a fade-in :class:`TokenSchedule`.

    Args:
        token_expr (str): Token expression to schedule.
        token_indices (Sequence[int] | Sequence[str]): Token indices resolved by
            the embedding backend.
        start_time (float): Normalised time where the fade starts.
        end_time (float): Normalised time where the fade ends.
        curve_type (CurveType | str | None, optional): Curve descriptor to use.
        **metadata (object): Arbitrary metadata attached to the schedule.

    Returns:
        TokenSchedule: Configured fade-in schedule.
    """

    params = ScheduleParams(start_time=start_time, end_time=end_time, curve_type=_normalize_curve(curve_type))
    return TokenSchedule(
        token_expr=token_expr,
        token_indices=list(token_indices),
        params=params,
        direction="fade_in",
        metadata=dict(metadata),
    )


def create_fade_out_schedule(
    token_expr: str,
    token_indices: Sequence[int] | Sequence[str],
    start_time: float,
    end_time: float,
    curve_type: CurveType | str | None = None,
    **metadata: object,
) -> TokenSchedule:
    """Convenience factory for a fade-out :class:`TokenSchedule`.

    Args:
        token_expr (str): Token expression to schedule.
        token_indices (Sequence[int] | Sequence[str]): Token indices resolved by
            the embedding backend.
        start_time (float): Normalised time where the fade starts.
        end_time (float): Normalised time where the fade ends.
        curve_type (CurveType | str | None, optional): Curve descriptor to use.
        **metadata (object): Arbitrary metadata attached to the schedule.

    Returns:
        TokenSchedule: Configured fade-out schedule.
    """

    params = ScheduleParams(start_time=start_time, end_time=end_time, curve_type=_normalize_curve(curve_type))
    return TokenSchedule(
        token_expr=token_expr,
        token_indices=list(token_indices),
        params=params,
        direction="fade_out",
        metadata=dict(metadata),
    )
