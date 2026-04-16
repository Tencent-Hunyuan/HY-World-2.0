import tempfile
import unittest
from pathlib import Path

import numpy as np

from hyworld2.worldrecon.hyworldmirror.utils.output_budget import build_output_save_plan

STREAMING_IMPORT_ERROR = None
try:
    import torch
    from plyfile import PlyData

    from hyworld2.worldrecon.hyworldmirror.utils.streaming_save import (
        _aggregate_gaussians_by_voxel,
        _finalize_gaussian_aggregates,
        save_gaussian_splats_artifact,
    )
except ModuleNotFoundError as exc:
    torch = None
    STREAMING_IMPORT_ERROR = exc


class OutputSavePlanTest(unittest.TestCase):
    def test_explicit_frame_chunk_controls_chunk_size(self):
        plan = build_output_save_plan(
            8, 10, 20,
            memory_budget_gb=1.0,
            save_chunk_frames=2,
        )

        self.assertEqual(plan.point_frame_chunk_size, 2)
        self.assertEqual(plan.gaussian_chunk_elements, 400)
        self.assertEqual(plan.total_pixels, 1600)

    def test_memory_budget_caps_gaussian_chunk_below_total(self):
        plan = build_output_save_plan(
            32, 952, 952,
            memory_budget_gb=0.01,
        )

        self.assertGreaterEqual(plan.gaussian_chunk_elements, 16_384)
        self.assertLess(plan.gaussian_chunk_elements, plan.total_pixels)
        self.assertGreaterEqual(plan.point_frame_chunk_size, 1)


@unittest.skipIf(torch is None, f"streaming save dependencies unavailable: {STREAMING_IMPORT_ERROR}")
class GaussianAggregationTest(unittest.TestCase):
    def test_chunked_voxel_merge_matches_single_pass_grouping(self):
        means = np.array(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        scales = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        quats = np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        colors = np.array(
            [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.9, 1.0, 1.1]],
            dtype=np.float32,
        )
        opacities = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        weights = np.array([1.0, 3.0, 2.0], dtype=np.float32)

        first = _aggregate_gaussians_by_voxel(
            means[:1], scales[:1], quats[:1], colors[:1], opacities[:1], weights[:1],
            voxel_size=0.01,
        )
        second = _aggregate_gaussians_by_voxel(
            means[1:], scales[1:], quats[1:], colors[1:], opacities[1:], weights[1:],
            voxel_size=0.01,
        )
        chunked = _finalize_gaussian_aggregates([first, second])

        expected_mean = np.array([[0.00075, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        expected_scale = np.array([[2.5, 3.5, 4.5], [7.0, 8.0, 9.0]], dtype=np.float32)
        expected_color = np.array([[0.4, 0.5, 0.6], [0.9, 1.0, 1.1]], dtype=np.float32)
        expected_opacity = np.array([2.5, 2.0], dtype=np.float32)

        np.testing.assert_allclose(chunked[1], expected_mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(chunked[2], expected_scale, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(chunked[4], expected_color, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(chunked[5], expected_opacity, rtol=1e-6, atol=1e-6)


@unittest.skipIf(torch is None, f"streaming save dependencies unavailable: {STREAMING_IMPORT_ERROR}")
class GaussianSaveTest(unittest.TestCase):
    def test_streaming_gaussian_writer_emits_valid_ply(self):
        splats = {
            "means": torch.tensor(
                [[[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            "scales": torch.ones((1, 1, 4, 3), dtype=torch.float32) * 0.01,
            "quats": torch.tensor(
                [[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            "opacities": torch.ones((1, 1, 4), dtype=torch.float32),
            "sh": torch.ones((1, 1, 4, 1, 3), dtype=torch.float32) * 0.5,
            "weights": torch.ones((1, 1, 4), dtype=torch.float32),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "gaussians.ply"
            timings = save_gaussian_splats_artifact(
                out_path,
                splats,
                image_shape=(1, 1, 4),
                save_chunk_frames=1,
                voxel_size=0.01,
            )
            ply = PlyData.read(str(out_path))

        self.assertEqual(timings["gs_source_count"], 4.0)
        self.assertEqual(len(ply["vertex"]), 3)
        self.assertIn("f_dc_0", ply["vertex"].data.dtype.names)
        self.assertIn("rot_3", ply["vertex"].data.dtype.names)


if __name__ == "__main__":
    unittest.main()
