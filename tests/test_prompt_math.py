import unittest
import sys
import types
import torch
from tensor_math.prompt_math.scheduling import (
    CurveType,
    ScheduleParams,
    TokenSchedule,
    CurveFunctions,
    TimestepConverter,
    ScheduleEvaluator,
    AttentionScheduler,
    create_fade_in_schedule,
    create_fade_out_schedule,
)
from tensor_math.prompt_math.parser import ExtendedParser
from tensor_math.prompt_math.evaluator import (
    evaluate_expr_with_scheduling,
    EvaluationContext,
    ScheduleFactory,
    build_frontend_config,
    get_pad_vector,
    discover_comfy_schedulers,
)
from tensor_math.prompt_math.functions import (
    VectorOperations,
    StatisticalOperations,
    MaskingOperations,
    NormalizationOperations,
)



def clear_comfy_stub():
    for name in [key for key in list(sys.modules.keys()) if key == 'comfy' or key.startswith('comfy.')]:
        sys.modules.pop(name, None)


def install_comfy_stub():
    clear_comfy_stub()
    comfy_pkg = types.ModuleType('comfy')
    comfy_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules['comfy'] = comfy_pkg

    samplers_mod = types.ModuleType('comfy.samplers')

    class Handler:  # pylint: disable=too-few-public-methods
        def __init__(self, handler, use_ms):
            self.handler = handler
            self.use_ms = use_ms

    def simple_scheduler(model_sampling, steps):  # pragma: no cover - stub
        return steps

    def karras_scheduler(n, sigma_min, sigma_max):  # pragma: no cover - stub
        return sigma_min + sigma_max

    samplers_mod.SCHEDULER_HANDLERS = {
        'simple': Handler(simple_scheduler, True),
        'karras': Handler(karras_scheduler, False),
    }
    samplers_mod.SCHEDULER_NAMES = ['simple', 'karras']

    class DummyKSampler:  # pylint: disable=too-few-public-methods
        SCHEDULERS = ['simple', 'karras']

    samplers_mod.KSampler = DummyKSampler
    sys.modules['comfy.samplers'] = samplers_mod


class TestScheduling(unittest.TestCase):
    def test_curve_functions(self):
        self.assertAlmostEqual(CurveFunctions.linear(0.5, 0, 1), 0.5)
        self.assertAlmostEqual(CurveFunctions.smooth(0.5, 0, 1), 0.5)
        self.assertAlmostEqual(CurveFunctions.ease_in(0.5, 0, 1), 0.25)
        self.assertAlmostEqual(CurveFunctions.ease_out(0.5, 0, 1), 0.75)

    def test_curve_functions_ease_in_out(self):
        self.assertAlmostEqual(CurveFunctions.ease_in_out(0.5, 0, 1), 0.5)

    def test_timestep_converter(self):
        self.assertAlmostEqual(
            TimestepConverter.to_normalized(10, "step_based", total_steps=20), 0.5
        )

    def test_schedule_params_validation(self):
        with self.assertRaises(ValueError):
            ScheduleParams(start_time=0.5, end_time=0.2)

    def test_schedule_evaluator(self):
        schedule = create_fade_in_schedule(
            "test", [0], start_time=0.2, end_time=0.8, curve_type=CurveType.LINEAR
        )
        evaluator = ScheduleEvaluator()
        self.assertAlmostEqual(evaluator.evaluate(schedule, 0.5), 0.5)

    def test_fade_out_schedule_weights(self):
        schedule = create_fade_out_schedule(
            "token",
            [0],
            start_time=0.0,
            end_time=1.0,
            curve_type=CurveType.SMOOTH,
        )
        self.assertIsInstance(schedule, TokenSchedule)
        self.assertGreater(schedule.weight_at(0.1), schedule.weight_at(0.9))

    def test_attention_scheduler_weight_progression(self):
        scheduler = AttentionScheduler()
        schedule = create_fade_in_schedule(
            "token", [0], start_time=0.0, end_time=1.0, curve_type=CurveType.SMOOTH
        )
        scheduler.register(schedule)
        self.assertLess(scheduler.weight_at("token", 0.2), scheduler.weight_at("token", 0.8))


class TestParser(unittest.TestCase):
    def test_basic_parsing(self):
        parser = ExtendedParser("[[[a]-[b]]]")
        ast = parser.parse()
        self.assertEqual(ast.kind, "op")

    def test_scheduling_parsing(self):
        parser = ExtendedParser("[[[a] @ fade_in(0.2, 0.8)]]")
        ast = parser.parse()
        self.assertIsNotNone(ast.schedule)
        self.assertEqual(ast.schedule.function_name, "fade_in")

    def test_parse_addition(self):
        parser = ExtendedParser("[[[a]+[b]]]")
        ast = parser.parse()
        self.assertEqual(ast.kind, "op")
        self.assertEqual(ast.value, "+")


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.a = torch.ones(4)
        self.b = torch.zeros(4)
        self.lookup = lambda text, **kwargs: self.a if text == "a" else self.b
        self.pad = lambda: torch.zeros(4)

    def test_basic_evaluation(self):
        parser = ExtendedParser("[[[a]-[b]]]")
        ast = parser.parse()
        result, _ = evaluate_expr_with_scheduling(ast, self.lookup, "clip_l", self.pad)
        self.assertTrue(torch.all(torch.eq(result, torch.ones(4))))

    def test_addition_evaluation(self):
        parser = ExtendedParser("[[[a]+[b]]]")
        ast = parser.parse()
        result, _ = evaluate_expr_with_scheduling(ast, self.lookup, "clip_l", self.pad)
        self.assertTrue(torch.all(torch.eq(result, torch.ones(4))))

    def test_scheduling_evaluation(self):
        scheduler = AttentionScheduler()
        context = EvaluationContext(
            encoder="clip_l", scheduler=scheduler, token_lookup=self.lookup, pad_fallback=self.pad
        )
        parser = ExtendedParser("[[[a] @ fade_in(0.2, 0.8)]]")
        ast = parser.parse()
        _, schedules = evaluate_expr_with_scheduling(
            ast, self.lookup, "clip_l", self.pad, context
        )
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].token_expr, "a")

    def test_schedule_factory_curve_selection(self):
        scheduler = AttentionScheduler()
        context = EvaluationContext(
            encoder="clip_l", scheduler=scheduler, token_lookup=self.lookup, pad_fallback=self.pad
        )
        parser = ExtendedParser('[[[a] @ fade_in(0.0, 1.0, "ease_in_out")]]')
        ast = parser.parse()
        _, schedules = evaluate_expr_with_scheduling(
            ast, self.lookup, "clip_l", self.pad, context
        )
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].params.curve_type, CurveType.EASE_IN_OUT)

    def test_schedule_factory_custom_registration(self):
        factory = ScheduleFactory()

        def custom_builder(token_expr, token_indices, call):
            return create_fade_out_schedule(token_expr, token_indices, 0.0, 1.0)

        factory.register("custom_fade", custom_builder)
        scheduler = AttentionScheduler()
        context = EvaluationContext(
            encoder="clip_l",
            scheduler=scheduler,
            token_lookup=self.lookup,
            pad_fallback=self.pad,
            schedule_factory=factory,
        )
        parser = ExtendedParser("[[[a] @ custom_fade(0.2, 0.8)]]")
        ast = parser.parse()
        _, schedules = evaluate_expr_with_scheduling(
            ast, self.lookup, "clip_l", self.pad, context
        )
        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].direction, "fade_out")

    def test_get_pad_vector_fallback(self):
        pad = lambda: torch.full((4,), 0.25)
        pad_vec = get_pad_vector(pad)
        self.assertTrue(torch.all(torch.eq(pad_vec, torch.full((4,), 0.25))))


class TestFactoryIntrospection(unittest.TestCase):
    def test_schedule_factory_metadata_export(self):
        factory = ScheduleFactory()
        metadata = factory.metadata()
        self.assertIn("fade_in", metadata)
        self.assertAlmostEqual(metadata["fade_in"]["defaults"]["start"], 0.0)
        config = build_frontend_config(factory)
        self.assertIn("scheduleFunctions", config)
        self.assertIn("fade_out", config["scheduleFunctions"])
        self.assertIn("templates", config)
        self.assertTrue(config["templates"])
        self.assertIn("samplers", config)
        self.assertIn("available", config["samplers"])
        self.assertIn("metadata", config["samplers"])


class TestSchedulerDiscovery(unittest.TestCase):
    def tearDown(self):
        clear_comfy_stub()

    def test_discovery_without_comfy(self):
        clear_comfy_stub()
        data = discover_comfy_schedulers()
        self.assertIn('source', data)
        self.assertEqual(data['available'], [])
        self.assertEqual(data['metadata'], {})

    def test_discovery_with_stub(self):
        install_comfy_stub()
        data = discover_comfy_schedulers()
        self.assertIn('simple', data['available'])
        self.assertIn('simple', data['metadata'])
        self.assertTrue(data['metadata']['simple']['useModelSampling'])
        config = build_frontend_config()
        self.assertIn('simple', config['samplers']['available'])

class TestExtendedFunctions(unittest.TestCase):
    def test_vector_operations(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = VectorOperations.lerp(a, b, 0.5)
        self.assertTrue(torch.all(torch.eq(result, torch.tensor([0.5, 0.5]))))

    def test_statistical_operations(self):
        vectors = [torch.ones(4), torch.zeros(4)]
        result = StatisticalOperations.weighted_mean(vectors, [0.5, 0.5])
        self.assertTrue(torch.all(torch.eq(result, torch.full((4,), 0.5))))

    def test_masking_operations(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        result = MaskingOperations.apply_mask(vector, mask, fill_value=-1.0)
        self.assertTrue(torch.all(torch.eq(result, torch.tensor([1.0, -1.0, 3.0]))))

    def test_layer_norm_normalizes(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        normalized = NormalizationOperations.layer_norm(vector)
        mean_val = torch.mean(normalized).item()
        centered = normalized - torch.mean(normalized)
        variance = torch.mean(centered ** 2).item()
        self.assertAlmostEqual(mean_val, 0.0, places=5)
        self.assertAlmostEqual(variance, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()









