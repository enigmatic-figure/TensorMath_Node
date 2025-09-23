"""Evaluation helpers for Prompt Math expressions with scheduling support.

The evaluator walks the parsed AST, materialises tensors via a lookup, and
registers attention schedules for downstream ComfyUI nodes.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import torch

from .parser import ASTNode, ScheduleCall
from .scheduling import (
    AttentionScheduler,
    CurveType,
    ScheduleEvaluator,
    TokenSchedule,
    create_fade_in_schedule,
    create_fade_out_schedule,
)

__all__ = [
    "EvaluationContext",
    "ScheduleFactory",
    "evaluate_expr_with_scheduling",
    "get_pad_vector",
    "DEFAULT_SCHEDULE_METADATA",
    "DEFAULT_TEMPLATES",
    "discover_comfy_schedulers",
    "build_frontend_config",
]


DEFAULT_SCHEDULE_METADATA: dict[str, dict[str, object]] = {
    "fade_in": {
        "label": "Fade In",
        "direction": "increase",
        "description": "Gradually raise attention between start and end timesteps.",
        "defaults": {"start": 0.0, "end": 1.0},
        "parameters": [
            {"name": "start", "type": "float", "required": True},
            {"name": "end", "type": "float", "required": True},
            {"name": "curve", "type": "string", "required": False},
        ],
        "template": "@ fade_in({start}, {end})",
    },
    "fade_out": {
        "label": "Fade Out",
        "direction": "decrease",
        "description": "Gradually lower attention between start and end timesteps.",
        "defaults": {"start": 0.0, "end": 1.0},
        "parameters": [
            {"name": "start", "type": "float", "required": True},
            {"name": "end", "type": "float", "required": True},
            {"name": "curve", "type": "string", "required": False},
        ],
        "template": "@ fade_out({start}, {end})",
    },
}


DEFAULT_TEMPLATES: List[dict[str, str]] = [
    {"name": "Basic Analogy", "code": "[[ [king] - [man] + [woman] ]]", "description": "Classic vector arithmetic analogy"},
    {"name": "Quality Aggregate", "code": "[[ mean([blurry],[grainy],[ugly]) ]]", "description": "Average multiple negative qualities"},
    {"name": "Style Transfer", "code": "[[ [content] + 0.7*([style] - mean([photo],[realistic])) ]]", "description": "Transfer style while preserving content"},
    {"name": "Temporal Fade In", "code": "[[ [detailed] @ fade_in(0.2, 0.8) ]]", "description": "Gradually introduce details during sampling"},
    {"name": "Style Morphing", "code": "[[ [oil_painting] @ fade_out(0.0, 0.5) + [watercolor] @ fade_in(0.5, 1.0) ]]", "description": "Transition from one style to another"},
    {"name": "Emphasis Burst", "code": "[[ [sharp] @ emphasis(2.0, 0.0, 0.3) ]]", "description": "Strong emphasis early in sampling"},
    {"name": "Bell Curve Focus", "code": "[[ [glowing] @ bell(0.5, 0.2, 2.0) ]]", "description": "Peak attention in the middle of sampling"},
    {"name": "Pulsing Effect", "code": "[[ [dynamic] @ pulse(3, 0.5) ]]", "description": "Oscillating attention throughout sampling"},
]


def _normalise_default(value: object) -> object:
    """Convert parameter defaults to JSON-serialisable structures.

    Args:
        value (object): Arbitrary default value.

    Returns:
        object: Normalised representation suitable for JSON encoding.
    """

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_normalise_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise_default(item) for key, item in value.items()}
    return str(value)


def _describe_parameters(callable_obj: Callable[..., object]) -> List[dict[str, object]]:
    """Extract a serialisable signature description for helper metadata.

    Args:
        callable_obj (Callable[..., object]): Callable to introspect.

    Returns:
        List[dict[str, object]]: Parameter descriptors.
    """

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return []

    parameters: List[dict[str, object]] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry: dict[str, object] = {"name": parameter.name, "kind": parameter.kind.name.lower()}
        if parameter.default is not inspect._empty:
            entry["default"] = _normalise_default(parameter.default)
        if parameter.annotation is not inspect._empty:
            entry["annotation"] = getattr(parameter.annotation, "__name__", str(parameter.annotation))
        parameters.append(entry)
    return parameters


def _resolve_scheduler_names(module: object, handler_names: Iterable[str]) -> List[str]:
    """Best-effort resolution of scheduler names from a ComfyUI module.

    Args:
        module (object): Imported comfy.samplers module.
        handler_names (Iterable[str]): Known handler keys.

    Returns:
        List[str]: Ordered list of unique scheduler names.
    """

    candidates: List[str] = []
    for attribute in ("SCHEDULERS", "SCHEDULER_NAMES"):
        value = getattr(module, attribute, None)
        if value:
            candidates.extend(list(value))

    ksampler = getattr(module, "KSampler", None)
    if ksampler is not None:
        kschedulers = getattr(ksampler, "SCHEDULERS", None)
        if kschedulers:
            candidates.extend(list(kschedulers))

    if not candidates:
        candidates.extend(list(handler_names))

    seen: set[str] = set()
    ordered: List[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _empty_scheduler_payload(source: str = "fallback") -> dict[str, object]:
    """Return an empty scheduler payload describing the retrieval status.

    Args:
        source (str): Reason or origin for the empty payload.

    Returns:
        dict[str, object]: Placeholder scheduler metadata.
    """

    return {"source": source, "available": [], "metadata": {}}


def discover_comfy_schedulers() -> dict[str, object]:
    """Discover scheduler metadata from comfy.samplers when available.

    Returns:
        dict[str, object]: Scheduler discovery payload with names and metadata.
    """

    try:
        import comfy.samplers as comfy_samplers  # type: ignore[import-not-found]  # noqa: WPS433
    except Exception:  # pragma: no cover - optional dependency
        return _empty_scheduler_payload()

    handler_map = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    if not handler_map:
        return _empty_scheduler_payload("missing_handlers")

    metadata: dict[str, dict[str, object]] = {}
    for name, handler in handler_map.items():
        handler_callable = getattr(handler, "handler", handler)
        use_model_sampling = bool(getattr(handler, "use_ms", getattr(handler, "use_model_sampling", False)))
        parameters = _describe_parameters(handler_callable)
        if not parameters:
            if use_model_sampling:
                parameters = [
                    {"name": "model_sampling", "kind": "positional"},
                    {"name": "steps", "kind": "positional"},
                ]
            else:
                parameters = [
                    {"name": "n", "kind": "positional"},
                    {"name": "sigma_min", "kind": "keyword"},
                    {"name": "sigma_max", "kind": "keyword"},
                ]

        metadata[name] = {
            "label": name.replace('_', ' ').title(),
            "useModelSampling": use_model_sampling,
            "parameters": parameters,
            "call": f"{name}(model_sampling, steps)" if use_model_sampling else f"{name}(n, sigma_min, sigma_max)",
            "module": getattr(handler_callable, "__module__", "comfy.samplers"),
        }

    available = _resolve_scheduler_names(comfy_samplers, metadata.keys())
    return {"source": "comfy.samplers", "available": available, "metadata": metadata}




def _as_float(value: object, fallback: float) -> float:
    """Best-effort float conversion with ``fallback`` for invalid inputs."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


class ScheduleFactory:
    """Factory that maps parsed schedule calls to runtime scheduling objects."""

    def __init__(self) -> None:
        """Initialise the registry with built-in fade helpers."""

        self._builders: dict[
            str, Callable[[str, Sequence[int] | Sequence[str], ScheduleCall], TokenSchedule]
        ] = {}
        self._metadata: dict[str, dict[str, object]] = {}
        self._register_default_builders()

    def _register_default_builders(self) -> None:
        self.register(
            "fade_in",
            self._build_fade_in,
            metadata=DEFAULT_SCHEDULE_METADATA.get("fade_in"),
        )
        self.register(
            "fade_out",
            self._build_fade_out,
            metadata=DEFAULT_SCHEDULE_METADATA.get("fade_out"),
        )

    def register(
        self,
        name: str,
        builder: Callable[[str, Sequence[int] | Sequence[str], ScheduleCall], TokenSchedule],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Register a schedule builder.

        Args:
            name (str): Name exposed to parser annotations.
            builder (Callable): Callable producing a :class:`TokenSchedule`.
            metadata (dict[str, object] | None): Optional metadata describing the
                schedule for front-end consumers.
        """

        self._builders[name] = builder
        if metadata is not None:
            self._metadata[name] = metadata
        elif name in DEFAULT_SCHEDULE_METADATA and name not in self._metadata:
            self._metadata[name] = DEFAULT_SCHEDULE_METADATA[name]

    def registered_functions(self) -> Tuple[str, ...]:
        """tuple[str, ...]: Names of all registered schedule functions."""

        return tuple(self._builders.keys())

    def metadata(self) -> dict[str, dict[str, object]]:
        """Return a deep copy of metadata for registered schedules."""

        available = {}
        for name in self._builders:
            if name in self._metadata:
                available[name] = deepcopy(self._metadata[name])
        return available

    def create(
        self,
        call: ScheduleCall,
        token_expr: str,
        token_indices: Sequence[int] | Sequence[str] | None = None,
    ) -> TokenSchedule:
        """Instantiate a schedule for ``token_expr``.

        Args:
            call (ScheduleCall): Parsed call descriptor.
            token_expr (str): Token expression owning the schedule.
            token_indices (Sequence[int] | Sequence[str] | None, optional):
                Optional embedding indices.

        Returns:
            TokenSchedule: Schedule produced by a registered builder.

        Raises:
            ValueError: If no builder is registered under ``call.function_name``.
        """

        builder = self._builders.get(call.function_name)
        if builder is None:
            raise ValueError(f"Unknown schedule function: {call.function_name}")
        return builder(token_expr, token_indices or [], call)

    def _build_fade_in(
        self,
        token_expr: str,
        token_indices: Sequence[int] | Sequence[str],
        call: ScheduleCall,
    ) -> TokenSchedule:
        """Internal helper that emits a fade-in schedule."""

        start, end, curve, metadata = self._extract_common_args(call)
        return create_fade_in_schedule(
            token_expr,
            token_indices,
            start,
            end,
            curve_type=curve,
            **metadata,
        )

    def _build_fade_out(
        self,
        token_expr: str,
        token_indices: Sequence[int] | Sequence[str],
        call: ScheduleCall,
    ) -> TokenSchedule:
        """Internal helper that emits a fade-out schedule."""

        start, end, curve, metadata = self._extract_common_args(call)
        return create_fade_out_schedule(
            token_expr,
            token_indices,
            start,
            end,
            curve_type=curve,
            **metadata,
        )

    def _extract_common_args(
        self, call: ScheduleCall
    ) -> tuple[float, float, CurveType | str | None, dict[str, object]]:
        """Extract standard start/end/curve arguments from ``call``."""

        args = list(call.args)
        start = _as_float(args[0] if args else call.kwargs.get("start", 0.0), 0.0)
        end = _as_float(args[1] if len(args) > 1 else call.kwargs.get("end", 1.0), 1.0)
        curve = call.kwargs.get("curve") or call.kwargs.get("curve_type")
        if curve is None and len(args) > 2:
            curve = args[2]
        metadata = {
            key: value
            for key, value in call.kwargs.items()
            if key not in {"start", "end", "curve", "curve_type"}
        }
        return float(start), float(end), curve, metadata


def build_frontend_config(factory: ScheduleFactory | None = None) -> dict[str, object]:
    """Construct a front-end configuration payload detailing schedules.

    Args:
        factory (ScheduleFactory | None): Optional factory instance to query.

    Returns:
        dict[str, object]: Serializable configuration containing schedule
            metadata, ComfyUI scheduler discovery data, and editor templates.
    """

    active_factory = factory or ScheduleFactory()
    return {
        "scheduleFunctions": active_factory.metadata(),
        "samplers": discover_comfy_schedulers(),
        "templates": deepcopy(DEFAULT_TEMPLATES),
    }


@dataclass
class EvaluationContext:
    """Aggregates reusable context for expression evaluation.

    Attributes:
        encoder (str): Name of the encoder requested by downstream nodes.
        scheduler (AttentionScheduler | None): Optional scheduler registry.
        token_lookup (Callable[..., torch.Tensor] | None): Token-to-tensor
            lookup.
        pad_fallback (Callable[[], torch.Tensor] | None): Fallback pad vector
            factory.
        schedule_factory (ScheduleFactory): Factory for schedule instantiation.
        auto_register (bool): Whether new schedules register themselves
            automatically.
    """

    encoder: str
    scheduler: AttentionScheduler | None = None
    token_lookup: Callable[..., torch.Tensor] | None = None
    pad_fallback: Callable[[], torch.Tensor] | None = None
    schedule_factory: ScheduleFactory = field(default_factory=ScheduleFactory)
    auto_register: bool = True

    def __post_init__(self) -> None:
        """Ensure a scheduler is always available."""

        if self.scheduler is None:
            self.scheduler = AttentionScheduler()


def get_pad_vector(
    pad_fallback: Callable[[], torch.Tensor] | None,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve a pad vector using a fallback or structural hint.

    Args:
        pad_fallback (Callable[[], torch.Tensor] | None): Callable producing a
            pad token when available.
        like (torch.Tensor | None, optional): Tensor providing shape hints when
            the fallback returns ``None``.

    Returns:
        torch.Tensor: Pad tensor matching the downstream expectations.

    Raises:
        ValueError: If no pad vector can be produced.
    """

    if pad_fallback is not None:
        pad = pad_fallback()
        if pad is not None:
            return pad
    if like is not None:
        return torch.zeros_like(like)
    raise ValueError("Unable to produce a pad vector; provide a pad_fallback callable.")


def evaluate_expr_with_scheduling(
    ast: ASTNode,
    token_lookup: Callable[..., torch.Tensor],
    encoder: str,
    pad_fallback: Callable[[], torch.Tensor],
    context: EvaluationContext | None = None,
) -> Tuple[torch.Tensor, List[TokenSchedule]]:
    """Evaluate an AST and return its tensor plus registered schedules.

    Args:
        ast (ASTNode): Parsed expression tree.
        token_lookup (Callable[..., torch.Tensor]): Callable resolving tokens to
            tensors. Receives ``token`` and ``encoder`` keyword arguments.
        encoder (str): Encoder identifier forwarded to ``token_lookup``.
        pad_fallback (Callable[[], torch.Tensor]): Callable producing pad
            tensors when lookups return ``None``.
        context (EvaluationContext | None, optional): Explicit evaluation
            context. When ``None`` a default context is created.

    Returns:
        Tuple[torch.Tensor, List[TokenSchedule]]: Result tensor and a list of
        schedules discovered during evaluation.

    Raises:
        ValueError: If unsupported AST node kinds or operators are encountered.
    """

    if context is None:
        ctx = EvaluationContext(
            encoder=encoder,
            scheduler=AttentionScheduler(),
            token_lookup=token_lookup,
            pad_fallback=pad_fallback,
        )
    else:
        ctx = context
        ctx.encoder = encoder
        if ctx.token_lookup is None:
            ctx.token_lookup = token_lookup
        if ctx.pad_fallback is None:
            ctx.pad_fallback = pad_fallback
        if ctx.scheduler is None:
            ctx.scheduler = AttentionScheduler()

    evaluator = ScheduleEvaluator()

    def _evaluate(node: ASTNode) -> tuple[torch.Tensor, List[TokenSchedule]]:
        """Recursively evaluate ``node`` and collect schedules."""

        if node.kind == "token":
            vector = ctx.token_lookup(node.value, encoder=ctx.encoder) if ctx.token_lookup else None
            if vector is None:
                vector = get_pad_vector(ctx.pad_fallback, like=None)
            schedules: List[TokenSchedule] = []
            if node.schedule is not None:
                schedule = ctx.schedule_factory.create(node.schedule, node.value)
                if ctx.auto_register and ctx.scheduler is not None:
                    ctx.scheduler.register(schedule)
                schedules.append(schedule)
            return vector, schedules
        if node.kind == "op":
            left, left_schedules = _evaluate(node.children[0])
            right, right_schedules = _evaluate(node.children[1])
            if node.value == "+":
                result = left + right
            elif node.value == "-":
                result = left - right
            elif node.value == "*":
                result = left * right
            else:  # pragma: no cover - defensive branch
                raise ValueError(f"Unsupported operator: {node.value}")
            return result, left_schedules + right_schedules
        raise ValueError(f"Unsupported AST node kind: {node.kind}")

    result, schedules = _evaluate(ast)
    if ctx.scheduler is not None and schedules:
        for schedule in schedules:
            evaluator.evaluate(schedule, schedule.params.start_time)
    return result, schedules


