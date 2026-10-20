import copy
import unittest
from unittest import mock

import torch
from torch import nn

from dmmd.config import ROOT, load_config
from dmmd.model import FeatureUNet
from dmmd.noise import DDPMNoise
from dmmd.objective import (
    discrepancy_from_statistics, feature_loss, linear_mmd2, linear_statistics,
)
from dmmd.runtime import frozen
from dmmd.sampling import sample_particles, witness_step
from train import validate_resume


class ScalarFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t):
        return self.scale * x.flatten(1)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_linear_estimator_matches_explicit_kernel_values_and_gradients(self):
        a = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
        aa, bb, ab = a @ a.T, b @ b.T, a @ b.T
        expected = ((aa.sum() - aa.diag().sum()) / 20
                    + (bb.sum() - bb.diag().sum()) / 42 - 2 * ab.mean())
        actual = linear_mmd2(a, b)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, (a, b), retain_graph=True)
        expected_grad = torch.autograd.grad(expected, (a, b))
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value)

    def test_off_diagonal_estimator_is_not_clamped(self):
        features = torch.tensor([[1.0], [-1.0]])
        self.assertLess(linear_mmd2(features, features).item(), 0)

    def test_pooled_statistics_match_full_batch_values_and_gradients(self):
        a = torch.randn(16, 5, dtype=torch.float64, requires_grad=True)
        b = torch.randn(16, 5, dtype=torch.float64, requires_grad=True)
        pooled = sum(linear_statistics(x, y)
                     for x, y in zip(a.chunk(4), b.chunk(4)))
        mmd, l2, difference = discrepancy_from_statistics(pooled, 5)
        expected_mmd = linear_mmd2(a, b)
        expected_l2 = a.square().sum(1).mean() + b.square().sum(1).mean()
        torch.testing.assert_close(mmd, expected_mmd)
        torch.testing.assert_close(l2, expected_l2)
        torch.testing.assert_close(difference, a.mean(0) - b.mean(0))
        actual_grad = torch.autograd.grad(mmd + 0.1 * l2, (a, b), retain_graph=True)
        expected_grad = torch.autograd.grad(expected_mmd + 0.1 * expected_l2, (a, b))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)

    def test_global_reduction_has_correct_gradient_scaling(self):
        # Simulate four identical ranks: SUM must run in backward as well.
        distributed_model, full_model = ScalarFeature(), ScalarFeature()
        distributed_model.scale.data.fill_(0.5)
        full_model.scale.data.fill_(0.5)
        clean = torch.zeros(4, 1, 1, 1)
        noisy = torch.ones_like(clean)

        def multiply_by_four(value, **kwargs):
            value.mul_(4)

        with mock.patch("dmmd.objective.dist.is_initialized", return_value=True), \
             mock.patch("dmmd.objective.dist.get_world_size", return_value=4), \
             mock.patch("dmmd.objective.dist.all_reduce", side_effect=multiply_by_four):
            distributed_loss, _ = feature_loss(
                distributed_model, clean, noisy, torch.tensor(0.5),
                mmd_scope="global",
            )
            distributed_loss.backward()
        full_loss, _ = feature_loss(
            full_model, clean.repeat(4, 1, 1, 1), noisy.repeat(4, 1, 1, 1),
            torch.tensor(0.5),
        )
        full_loss.backward()
        torch.testing.assert_close(distributed_loss, full_loss)
        torch.testing.assert_close(distributed_model.scale.grad, full_model.scale.grad)

    def test_four_gpu_recipe_keeps_global_batch_and_full_budget(self):
        config = load_config(ROOT / "configs/cifar10_4gpu.yaml")
        self.assertEqual(config["train"]["batch_size"] * 4, 64)
        self.assertEqual(config["train"]["mmd_scope"], "global")
        self.assertEqual(config["train"]["noise_levels_per_step"], 16)
        self.assertEqual(config["train"]["steps"], 250000)
        self.assertEqual(config["model"]["base_channels"], 128)
        self.assertEqual(config["data"]["validation_size"], 0)

    def test_too_small_mmd_batch_rejected(self):
        with self.assertRaises(ValueError):
            linear_mmd2(torch.zeros(1, 2), torch.zeros(2, 2))

    def test_feature_objective_maximizes_not_minimizes_discrepancy(self):
        model = ScalarFeature()
        clean, noisy = torch.ones(4, 1), torch.zeros(4, 1)
        before = linear_mmd2(model(clean, 0), model(noisy, 0)).item()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, _ = feature_loss(model, clean, noisy, torch.tensor(0.5), 0, 0)
        loss.backward()
        optimizer.step()
        after = linear_mmd2(model(clean, 0), model(noisy, 0)).item()
        self.assertGreater(after, before)

    def test_gradient_penalty_has_parameter_gradients(self):
        model = ScalarFeature()
        model.scale.data.fill_(0.5)
        clean = torch.zeros(4, 1, 1, 1)
        noisy = torch.ones_like(clean)
        loss, metrics = feature_loss(model, clean, noisy, torch.tensor(0.5), 0, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(model.scale.grad))
        # MMD=s^2 and GP=(s^2-1)^2, so d[-MMD+GP]/ds=-2.5 at s=0.5.
        # Detaching the reference feature mean would give the wrong derivative.
        torch.testing.assert_close(model.scale.grad, torch.tensor(-2.5))
        self.assertGreater(metrics["gp"].item(), 0)

    def test_common_time_required_for_a_comparison(self):
        with self.assertRaises(ValueError):
            feature_loss(ScalarFeature(), torch.zeros(4, 1), torch.ones(4, 1),
                         torch.linspace(0, 1, 4), 0, 0)

    def test_noise_schedule_endpoints_and_finite_gradients(self):
        noise = DDPMNoise()
        clean = torch.randn(3, 3, 32, 32, requires_grad=True)
        eps = torch.randn_like(clean)
        torch.testing.assert_close(noise(clean, 0.0, eps), clean)
        alpha, sigma = noise.coefficients(1.0)
        self.assertLess(alpha.item(), 0.01)
        self.assertGreater(sigma.item(), 0.99)
        noise(clean, 0.5, eps).sum().backward()
        self.assertTrue(torch.isfinite(clean.grad).all())
        with self.assertRaises(ValueError):
            noise(clean, 1.1)

    def test_sampler_direction_and_no_batch_size_divisor(self):
        model = ScalarFeature()
        with frozen(model):
            particles = torch.tensor([[2.0], [4.0]])
            updated, gradient = witness_step(
                model, particles, 0.5, torch.tensor([1.0]), 0.1,
                noisy_mean=torch.tensor([3.0]),
            )
            torch.testing.assert_close(gradient, torch.full_like(particles, 2))
            torch.testing.assert_close(updated, particles - 0.2)
            self.assertIsNone(model.scale.grad)

    def test_independent_sampling_is_unchanged_by_companion_particles(self):
        model = ScalarFeature()
        points = torch.tensor([[2.0], [4.0], [-1.0]])
        times = torch.tensor([0.9, 0.1])
        clean = torch.tensor([[1.0], [1.5]])
        noisy = torch.tensor([[2.0], [2.0]])
        with frozen(model):
            full = sample_particles(model, points, times, clean, 2, 0.1, noisy)
            single = sample_particles(model, points[:1], times, clean, 2, 0.1, noisy)
        torch.testing.assert_close(full[:1], single)

    def test_interacting_sampling_updates_means_between_steps(self):
        model = ScalarFeature()
        with frozen(model):
            output = sample_particles(
                model, torch.tensor([[2.0], [4.0]]), torch.tensor([0.5]),
                torch.tensor([[0.0]]), 2, 0.1,
            )
        torch.testing.assert_close(output, torch.tensor([[1.43], [3.43]]))

    def test_unet_supports_input_and_second_order_parameter_gradients(self):
        model = FeatureUNet(base_channels=8, channel_mult=[1, 2],
                            time_dim=16, head_dim=4, dropout=0)
        clean, noisy = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        loss, _ = feature_loss(model, clean, noisy, torch.tensor(0.3))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(model(clean, 0.5).shape, (4, 16))

    def test_eval_unet_does_not_mix_images_within_a_batch(self):
        model = FeatureUNet(base_channels=8, channel_mult=[1, 2],
                            time_dim=16, head_dim=4, dropout=0).eval()
        images = torch.randn(4, 3, 32, 32)
        torch.testing.assert_close(model(images[:1], 0.5), model(images, 0.5)[:1])

    def test_frozen_context_restores_flags_and_mode(self):
        model = ScalarFeature().train()
        with frozen(model):
            self.assertFalse(model.training)
            self.assertFalse(model.scale.requires_grad)
        self.assertTrue(model.training)
        self.assertTrue(model.scale.requires_grad)

    def test_config_inheritance_and_invalid_keys(self):
        config = load_config(ROOT / "configs/smoke.yaml")
        self.assertEqual(config["train"]["lr"], 0.0001)
        self.assertEqual(config["train"]["steps"], 2)
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs/pilot.yaml", ["train.typo=1"])
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs/pilot.yaml", ["train.batch_size=1"])

    def test_resume_protects_the_scientific_recipe(self):
        old = load_config(ROOT / "configs/pilot.yaml")
        new = copy.deepcopy(old)
        new["train"]["steps"] *= 2
        validate_resume(old, new)
        new["noise"]["beta_end"] = 0.03
        with self.assertRaises(ValueError):
            validate_resume(old, new)


if __name__ == "__main__":
    unittest.main()
