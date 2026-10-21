import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

import compare_heads as comparison
from dmmd.config import ROOT, load_config
from dmmd.model import FeatureUNet
from dmmd.runtime import file_sha256


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dmmd_comparison_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = load_config(ROOT / "configs/smoke.yaml")
        self.config["data"]["synthetic"] = False
        self.config["train"]["mmd_scope"] = "global"

    def fixture(self, label, grid, *, legacy=False, lr=None):
        config = copy.deepcopy(self.config)
        config["model"]["head_spatial_size"] = grid
        if lr is not None:
            config["train"]["lr"] = lr
        if legacy:
            config["model"].pop("head_spatial_size")
        checkpoint, stats = self.directory / (label+".pt"), self.directory / (label+"_stats.pt")
        model = FeatureUNet(**config["model"])
        torch.save({"kind": "dmmd_features", "step": 4000, "world_size": 1,
                    "config": config, "model": model.state_dict()}, checkpoint)
        settings = dict(config["sampling"], pixel_clip=None)
        torch.save({"kind": "dmmd_statistics", "checkpoint_sha256": file_sha256(checkpoint),
                    "model_config": config["model"], "noise_config": config["noise"],
                    "reference_size": 50000, "times": torch.linspace(.9, .1, 10),
                    "mean_clean": torch.zeros(10, model.output_dim),
                    "mean_noisy": torch.zeros(10, model.output_dim),
                    "sampling_config": settings, "synthetic": False}, stats)
        return checkpoint, stats

    def write_samples(self, directory, info, count=4, seed=100042):
        images = directory / "images"
        images.mkdir(parents=True)
        metadata = {
            "checkpoint_sha256": info["checkpoint_sha256"], "checkpoint_step": 4000,
            "num_samples": count, "seed": seed, "mode": "interacting",
            "reference_size": 50000, "synthetic": False, "times": info["times"],
            "pixel_updates_per_sample": 100,
            "png_export": "clamp raw pixels to [-1,1], then map to [0,1]",
            "sampling": {"particles": 200, "steps_per_level": 10,
                         "step_size": .1, "pixel_clip": None},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))
        for index in range(count):
            Image.new("RGB", (32, 32)).save(images / f"{index:06d}.png")

    def write_fid(self, directory, count, value):
        (directory / "fid.json").write_text(json.dumps({
            "generated_samples": count,
            "reference": "all 50000 unaugmented CIFAR-10 training images",
            "protocol": "uint8 PNG -> torch-fidelity default Inception features",
            "metrics": {"frechet_inception_distance": value},
        }))

    def argv(self, baseline, baseline_stats, spatial, spatial_stats):
        return ["compare_heads.py", "--baseline-checkpoint", str(baseline),
                "--baseline-stats", str(baseline_stats), "--spatial-checkpoint", str(spatial),
                "--spatial-stats", str(spatial_stats), "--num-samples", "4", "--device", "cpu",
                "--output-root", str(self.directory / "outputs")]

    def test_old_and_spatial_configs_match_except_head(self):
        a, sa = self.fixture("old", 1, legacy=True)
        b, sb = self.fixture("new", 4)
        baseline = comparison.inspect_checkpoint(a, sa, 1, 4000)
        spatial = comparison.inspect_checkpoint(b, sb, 4, 4000)
        self.assertEqual(baseline["recipe"], spatial["recipe"])
        self.assertNotEqual(baseline["checkpoint_sha256"], spatial["checkpoint_sha256"])

    def test_wrong_checkpoint_step_grid_and_stale_stats_rejected(self):
        a, sa = self.fixture("old", 1)
        with self.assertRaisesRegex(ValueError, "step"):
            comparison.inspect_checkpoint(a, sa, 1, 10000)
        with self.assertRaisesRegex(ValueError, "grid"):
            comparison.inspect_checkpoint(a, sa, 4, 4000)
        with a.open("ab") as handle:
            handle.write(b"changed checkpoint hash")
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            comparison.inspect_checkpoint(a, sa, 1, 4000)

    def test_reference_count_time_grid_and_synthetic_stats_rejected(self):
        for field, value in [("reference_size", 128), ("times", torch.tensor([.9,.1])),
                             ("synthetic", True)]:
            with self.subTest(field=field):
                a, sa = self.fixture(field, 1)
                stats = torch.load(sa, weights_only=False)
                stats[field] = value
                torch.save(stats, sa)
                with self.assertRaises(ValueError):
                    comparison.inspect_checkpoint(a, sa, 1, 4000)

    def test_samples_must_match_seed_protocol_and_all_filenames(self):
        a, sa = self.fixture("old", 1)
        info = comparison.inspect_checkpoint(a, sa, 1, 4000)
        directory = self.directory / "samples"
        self.write_samples(directory, info)
        comparison.validate_samples(directory, info, 4, 100042)
        with self.assertRaisesRegex(ValueError, "seed"):
            comparison.validate_samples(directory, info, 4, 2)
        (directory / "images/000002.png").unlink()
        with self.assertRaisesRegex(ValueError, "PNGs"):
            comparison.validate_samples(directory, info, 4, 100042)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            comparison.validate_samples(self.directory / "incomplete", info, 4, 100042)

    def test_metric_count_reference_and_nonfinite_values_rejected(self):
        directory = self.directory / "metrics"
        directory.mkdir()
        self.write_fid(directory, 4, 99.)
        self.assertEqual(comparison.read_fid(directory, 4), 99.)
        with self.assertRaisesRegex(ValueError, "count"):
            comparison.read_fid(directory, 5)
        self.write_fid(directory, 4, float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            comparison.read_fid(directory, 4)

    def test_validate_only_does_not_require_fid_or_launch_jobs(self):
        a, sa = self.fixture("old", 1, legacy=True)
        b, sb = self.fixture("new", 4)
        with mock.patch.object(sys, "argv", self.argv(a,sa,b,sb)+["--validate-only"]), \
             mock.patch.object(comparison.subprocess, "run") as run, \
             mock.patch.object(comparison.importlib.util, "find_spec") as find, \
             contextlib.redirect_stdout(io.StringIO()):
            comparison.main()
        run.assert_not_called()
        find.assert_not_called()

    def test_training_recipe_mismatch_stops_before_sampling(self):
        a, sa = self.fixture("old", 1)
        b, sb = self.fixture("new", 4, lr=.002)
        with mock.patch.object(sys, "argv", self.argv(a,sa,b,sb)), \
             mock.patch.object(comparison.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "Training recipes"):
                comparison.main()
        run.assert_not_called()

    def test_full_orchestration_and_sample_reuse_with_mocked_metrics(self):
        # Exercise orchestration only. These fabricated numbers are NOT FID measurements.
        a, sa = self.fixture("old", 1, legacy=True)
        b, sb = self.fixture("new", 4)
        infos = {str(a): comparison.inspect_checkpoint(a,sa,1,4000),
                 str(b): comparison.inspect_checkpoint(b,sb,4,4000)}
        def child(command, **kwargs):
            def argument(name):
                return command[command.index(name)+1]
            if Path(command[2]).name == "sample.py":
                self.assertEqual(argument("--particles"), "200")
                self.assertEqual(argument("--steps-per-level"), "10")
                self.assertEqual(argument("--seed"), "100042")
                self.write_samples(Path(argument("--output")), infos[argument("--checkpoint")])
            else:
                folder = Path(argument("--samples"))
                self.assertIn("--cpu", command)
                self.write_fid(folder, 4, 100. if folder.name == "global1" else 99.)
            return subprocess.CompletedProcess(command, 0)
        for expected_calls in (4, 2):
            with mock.patch.object(sys, "argv", self.argv(a,sa,b,sb)), \
                 mock.patch.object(comparison.importlib.util, "find_spec", return_value=object()), \
                 mock.patch.object(comparison.subprocess, "run", side_effect=child) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            self.assertEqual(run.call_count, expected_calls)
        summary = json.loads((self.directory / "outputs/comparison.json").read_text())
        self.assertEqual(summary["spatial_minus_global_fid"], -1.)
        self.assertEqual(len(summary["results"]), 2)


if __name__ == "__main__":
    unittest.main()
