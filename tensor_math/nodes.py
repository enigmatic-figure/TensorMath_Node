"""ComfyUI node implementations for TensorMath."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from tensor_math.prompt_math import (
    AttentionScheduler,
    EvaluationContext,
    ExtendedParser,
    ScheduleFactory,
    TokenSchedule,
    build_frontend_config,
    evaluate_expr_with_scheduling,
)


def _coerce_tensor(value: Any) -> torch.Tensor:
    """Best-effort conversion of `value` into a tensor."""

    if isinstance(value, torch.Tensor):
        return value
    if callable(value):
        return _coerce_tensor(value())
    if isinstance(value, dict):
        for key in ("tensor", "value", "embedding"):
            if key in value:
                return _coerce_tensor(value[key])
    try:
        return torch.tensor(value)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TypeError(
            "Token vectors must be torch tensors, callables returning tensors, or tensor-compatible sequences."
        ) from exc


def _serialise_schedule(schedule: TokenSchedule) -> Dict[str, Any]:
    """Convert a :class:TokenSchedule into JSON-friendly metadata."""

    params = schedule.params
    curve = getattr(params.curve_type, "value", str(params.curve_type))
    return {
        "token": schedule.token_expr,
        "direction": schedule.direction,
        "start": params.start_time,
        "end": params.end_time,
        "curve": curve,
        "clamp_output": params.clamp_output,
        "indices": list(schedule.token_indices),
        "metadata": dict(schedule.metadata),
    }


class PromptMathEvaluate:
    """Evaluate Prompt Math expressions against a token tensor library."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"multiline": True, "default": "[[ [token] ]]"}),
                "token_vectors": ("DICT",),
            },
            "optional": {
                "encoder": ("STRING", {"default": "clip_l"}),
                "pad_token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("TENSOR", "DICT")
    RETURN_NAMES = ("tensor", "schedule_payload")
    FUNCTION = "evaluate"
    CATEGORY = "conditioning/tensor math"

    @staticmethod
    def _prepare_library(token_vectors: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Normalise and validate the provided token tensor library."""

        if not token_vectors:
            raise ValueError("token_vectors must contain at least one entry.")
        prepared: Dict[str, torch.Tensor] = {}
        for key, value in token_vectors.items():
            prepared[key] = _coerce_tensor(value)
        return prepared

    def evaluate(
        self,
        expression: str,
        token_vectors: Dict[str, Any],
        encoder: str = "clip_l",
        pad_token: str = "",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        library = self._prepare_library(token_vectors)
        parser = ExtendedParser(expression)
        ast = parser.parse()
        sample_tensor = next(iter(library.values()))

        def token_lookup(token: str, *, encoder: str) -> torch.Tensor:
            try:
                return library[token]
            except KeyError as exc:
                raise ValueError(f"Missing tensor for token {token!r}.") from exc

        def pad_fallback() -> torch.Tensor:
            if pad_token:
                try:
                    return token_lookup(pad_token, encoder=encoder)
                except ValueError as exc:
                    raise ValueError(
                        f"Pad token {pad_token!r} not present in token_vectors. Provide a valid pad token or omit it."
                    ) from exc
            return torch.zeros_like(sample_tensor)

        context = EvaluationContext(
            encoder=encoder,
            scheduler=AttentionScheduler(),
            token_lookup=token_lookup,
            pad_fallback=pad_fallback,
            schedule_factory=ScheduleFactory(),
        )

        result, schedules = evaluate_expr_with_scheduling(ast, token_lookup, encoder, pad_fallback, context)

        schedule_payload = {
            "encoder": encoder,
            "schedules": [_serialise_schedule(schedule) for schedule in schedules],
        }
        return result, schedule_payload


class PromptMathFrontendConfig:
    """Expose the front-end configuration used by the custom widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "conditioning/tensor math"

    def build(self) -> Tuple[Dict[str, Any]]:
        return (build_frontend_config(),)


NODE_CLASS_MAPPINGS = {
    "PromptMathEvaluate": PromptMathEvaluate,
    "PromptMathFrontendConfig": PromptMathFrontendConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptMathEvaluate": "Prompt Math - Evaluate",
    "PromptMathFrontendConfig": "Prompt Math - Frontend Config",
}


