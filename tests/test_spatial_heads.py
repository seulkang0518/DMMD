import copy
import unittest

import torch
from torch.nn import functional as F

from dmmd.config import ROOT, load_config, validate
from dmmd.model import FeatureHead, FeatureUNet
from dmmd.objective import feature_loss
from dmmd.runtime import frozen
from dmmd.sampling import witness_step
from train import validate_resume


class SpatialHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def test_global_head_retains_exact_original_operation(self):
        head = FeatureHead(8, 4)
        images = torch.randn(3, 8, 8, 8)
        expected = head.projection(F.silu(head.norm(images)).mean((2, 3)))
        self.assertTrue(torch.equal(head(images), expected))
        self.assertEqual(head.projection.weight.shape, (4, 8))

    def test_spatial_head_distinguishes_arrangements_lost_by_global_pooling(self):
        global_head, spatial_head = FeatureHead(8, 2), FeatureHead(8, 2, 4)
        images = torch.randn(2, 8, 8, 8)
        shifted = images.roll(4, dims=3)
        torch.testing.assert_close(global_head(images), global_head(shifted))
        with torch.no_grad():
            spatial_head.projection.weight.zero_()
            spatial_head.projection.bias.zero_()
            spatial_head.projection.weight[0, 0] = 1
        self.assertGreater((spatial_head(images)-spatial_head(shifted)).abs().max().item(), 1e-4)

    def test_spatial_variant_preserves_backbone_initialization_and_feature_size(self):
        kwargs = dict(base_channels=8, channel_mult=[1, 2], time_dim=16, head_dim=4, dropout=0)
        torch.manual_seed(42)
        global_model = FeatureUNet(**kwargs)
        torch.manual_seed(42)
        spatial_model = FeatureUNet(**kwargs, head_spatial_size=4)
        global_parameters = global_model.state_dict()
        for name, parameter in spatial_model.state_dict().items():
            if not name.startswith("heads."):
                torch.testing.assert_close(parameter, global_parameters[name], rtol=0, atol=0)
        images = torch.randn(2, 3, 32, 32)
        self.assertEqual(global_model(images, .5).shape, (2, 16))
        self.assertEqual(spatial_model(images, .5).shape, (2, 16))

    def test_spatial_model_supports_full_gradient_penalty_and_sampling(self):
        model = FeatureUNet(base_channels=8, channel_mult=[1, 2], time_dim=16,
                            head_dim=4, dropout=.2, head_spatial_size=4)
        clean, noisy = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
        loss, _ = feature_loss(model, clean, noisy, torch.tensor(.5))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        with frozen(model):
            reference = model(clean, .5).mean(0).detach()
            moved, gradient = witness_step(model, noisy, .5, reference, .1)
        self.assertTrue(torch.isfinite(moved).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_legacy_config_can_load_and_resume_global_model(self):
        new = load_config(ROOT / "configs/pilot.yaml")
        old = copy.deepcopy(new)
        old["model"].pop("head_spatial_size")
        validate(old)  # Used by cache_stats.py on old checkpoint configs.
        validate_resume(old, new)
        before = FeatureUNet(**old["model"]).eval()
        after = FeatureUNet(**new["model"]).eval()
        after.load_state_dict(before.state_dict(), strict=True)
        images = torch.randn(2, 3, 32, 32)
        self.assertTrue(torch.equal(before(images, .5), after(images, .5)))
        new["model"]["head_spatial_size"] = 4
        with self.assertRaisesRegex(ValueError, "model"):
            validate_resume(old, new)

    def test_ablation_changes_only_head_and_run_controls(self):
        baseline = load_config(ROOT / "configs/paper_inspired.yaml", [
            "train.steps=10000", "train.mmd_scope=global", "train.log_every=10",
        ])
        ablation = load_config(ROOT / "configs/cifar10_spatial4_1gpu_10k.yaml")
        self.assertEqual(ablation["model"]["head_spatial_size"], 4)
        self.assertEqual(ablation["train"]["batch_size"], 64)
        self.assertEqual(ablation["train"]["noise_levels_per_step"], 16)
        self.assertEqual(ablation["runtime"]["expected_world_size"], 1)
        expected = copy.deepcopy(baseline)
        expected["model"]["head_spatial_size"] = 4
        expected["train"]["keep_last"] = 10
        expected["runtime"]["expected_world_size"] = 1
        self.assertEqual(ablation, expected)

    def test_invalid_head_grids_rejected(self):
        for size in (0, -1, True, 1.5, 8):
            with self.assertRaisesRegex(ValueError, "head_spatial_size"):
                load_config(ROOT / "configs/paper_inspired.yaml", [
                    f"model.head_spatial_size={size}",
                ])
        with self.assertRaises(ValueError):
            FeatureHead(8, 4, 0)
        with self.assertRaises(ValueError):
            FeatureHead(8, 4, 4)(torch.randn(2, 8, 2, 2))


if __name__ == "__main__":
    unittest.main()
