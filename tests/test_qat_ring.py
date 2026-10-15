import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostmpc.core.model import Manifest, Op, TensorRef, digest
from ghostmpc.eval.reference import evaluate_numpy

try:
    import tensorflow as tf

    from ghostmpc.training.qat_ring import RingQATModel
    _HAVE_TF = True
except Exception:  # pragma: no cover - TF is an optional extra
    _HAVE_TF = False

_MODEL_R32 = Path(__file__).resolve().parents[1] / "artifacts" / "model" / "manifest.json"
FP = (6, 15)  # ring32/f6 deployment profile
# ring16/f4 comm-reduction profile (k = ring_bits/2-1); the forward must stay
# bit-exact against the faithful oracle at this narrower ring before any GPU
# hour is spent, exactly as Q2 certified ring32/f6.
FP_R16 = (4, 7)


def _oracle_ints(captured: dict, f: int) -> dict:
    out = {}
    for name, value in captured.items():
        if any(name.endswith(s) for s in (":raw", ":acc", ":product")):
            continue
        out[name] = np.rint(np.asarray(value, dtype=np.float64) * (1 << f)
                            ).astype(np.int64)
    return out


def _tf_ints(captured: dict, f: int) -> dict:
    return {name: np.rint(tensor.numpy().astype(np.float64) * (1 << f)
                          ).astype(np.int64)
            for name, tensor in captured.items()
            if not any(name.endswith(s) for s in (":raw", ":acc", ":product"))}


@unittest.skipUnless(_HAVE_TF, "tensorflow extra not installed")
class RingQATFidelityTests(unittest.TestCase):
    def _toy(self, root: Path) -> Path:
        # conv -> fused PReLU -> GAP: exercises the raw producer, the fused
        # slope selection and truncation, and the pooling reciprocal.
        rng = np.random.default_rng(4)
        weights = np.concatenate([
            rng.uniform(-0.4, 0.4, size=27),   # conv kernel 1x1x3x3
            rng.uniform(-0.2, 0.2, size=3),    # conv bias
            rng.uniform(-0.5, 0.5, size=3),    # prelu alpha
        ]).astype(np.float64)
        np.save(root / "weights.npy", weights)
        manifest = Manifest(
            2, "toy", "url", "c", [2, 2, 3], [3],
            [Op("conv", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"]),
             Op("act", "PReLU", ["conv"], [2, 2, 3], {"shared_axes": [1, 2]},
                ["act:0"]),
             Op("pool", "GlobalAveragePooling2D", ["act"], [3])],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
             TensorRef("conv:1", [3], 9, 3),
             TensorRef("act:0", [3], 12, 3)], "weights.npy", "w",
            overflow_bounds={"conv": 4.0})
        manifest.save(root / "manifest.json")
        return root / "manifest.json"

    def _toy_bn(self, root: Path) -> Path:
        # conv (truncated) -> BatchNormalization -> GAP: exercises the folded
        # affine path and its round-trip through export_qat_manifest.
        rng = np.random.default_rng(7)
        weights = np.concatenate([
            rng.uniform(-0.4, 0.4, size=9),    # conv kernel 1x1x3x3
            rng.uniform(-0.2, 0.2, size=3),    # conv bias
            rng.uniform(0.5, 1.5, size=3),     # bn gamma
            rng.uniform(-0.3, 0.3, size=3),    # bn beta
            rng.uniform(-0.3, 0.3, size=3),    # bn moving mean
            rng.uniform(0.5, 1.5, size=3),     # bn moving variance
        ]).astype(np.float64)
        np.save(root / "weights.npy", weights)
        manifest = Manifest(
            2, "toybn", "url", "c", [2, 2, 3], [3],
            [Op("conv", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"]),
             Op("bn", "BatchNormalization", ["conv"], [2, 2, 3],
                {"epsilon": 1e-3, "scale": True, "center": True},
                ["bn:0", "bn:1", "bn:2", "bn:3"]),
             Op("pool", "GlobalAveragePooling2D", ["bn"], [3])],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
             TensorRef("conv:1", [3], 9, 3),
             TensorRef("bn:0", [3], 12, 3), TensorRef("bn:1", [3], 15, 3),
             TensorRef("bn:2", [3], 18, 3), TensorRef("bn:3", [3], 21, 3)],
            "weights.npy", "w", overflow_bounds={"conv": 4.0})
        manifest.save(root / "manifest.json")
        return root / "manifest.json"

    def test_export_roundtrip_is_oracle_identical(self):
        # An untrained student's exported weights must reproduce the oracle
        # bit-for-bit, including the BatchNorm gamma/beta/mean/var reconstruction.
        from ghostmpc.training.qat_train import export_qat_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy_bn(root)
            images = np.random.default_rng(5).uniform(-0.5, 0.5,
                                                      size=(3, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=6, total_bits=15,
                                 ring_bits=32, dtype=tf.float64)
            exported = export_qat_manifest(model, path, root / "exported")
            exported_manifest = Manifest.load(exported)
            self.assertEqual(
                exported_manifest.weights_sha256,
                digest(exported.parent / exported_manifest.weights_file))
            exported_weights = np.load(
                exported.parent / exported_manifest.weights_file)
            for ref in exported_manifest.tensors:
                values = exported_weights[
                    ref.offset:ref.offset + ref.length]
                self.assertEqual(ref.minimum, float(values.min()))
                self.assertEqual(ref.maximum, float(values.max()))
            original = evaluate_numpy(path, images, fixed_point=FP, ring_bits=32)
            roundtrip = evaluate_numpy(exported, images, fixed_point=FP,
                                       ring_bits=32)
            np.testing.assert_array_equal(
                np.rint(np.asarray(original) * 64).astype(np.int64),
                np.rint(np.asarray(roundtrip) * 64).astype(np.int64))

    def test_auto_feature_ops_picks_adds(self):
        from ghostmpc.core.model import Manifest as _M
        from ghostmpc.training.qat_train import _auto_feature_ops

        with tempfile.TemporaryDirectory() as directory:
            path = self._toy(Path(directory))
            manifest = _M.load(path)
            self.assertEqual(_auto_feature_ops(manifest), [])  # no Add ops

    def test_toy_is_bit_exact_against_the_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy(root)
            images = np.random.default_rng(9).uniform(-1, 1, size=(3, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=6, total_bits=15,
                                 ring_bits=32, dtype=tf.float64)
            _, captured = model(images, capture=True)
            _, oracle = evaluate_numpy(path, images, capture=True,
                                       fixed_point=FP, ring_bits=32)
            got = _tf_ints(captured, 6)
            want = _oracle_ints(oracle, 6)
            for name, target in want.items():
                self.assertIn(name, got)
                np.testing.assert_array_equal(
                    got[name], target, err_msg=f"op {name} diverges")

    def test_toy_is_bit_exact_at_ring16_f4(self):
        # Same fidelity guarantee at the ring16/f4 profile: the QAT forward and
        # the faithful oracle must agree op-for-op on the 2^f grid.
        f, k = FP_R16
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy(root)
            images = np.random.default_rng(9).uniform(-1, 1, size=(3, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=f, total_bits=k,
                                 ring_bits=16, dtype=tf.float64)
            _, captured = model(images, capture=True)
            _, oracle = evaluate_numpy(path, images, capture=True,
                                       fixed_point=FP_R16, ring_bits=16)
            got = _tf_ints(captured, f)
            want = _oracle_ints(oracle, f)
            for name, target in want.items():
                self.assertIn(name, got)
                np.testing.assert_array_equal(
                    got[name], target, err_msg=f"op {name} diverges at ring16")

    def test_gradients_flow_at_ring16_f4(self):
        f, k = FP_R16
        with tempfile.TemporaryDirectory() as directory:
            path = self._toy(Path(directory))
            images = np.random.default_rng(2).uniform(-1, 1, size=(4, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=f, total_bits=k,
                                 ring_bits=16, dtype=tf.float32)
            # The float32 forward must be certified exact for this profile,
            # otherwise train_qat_ring refuses to run (e_max + 2f <= 24).
            self.assertTrue(model._exact_float32)
            with tf.GradientTape() as tape:
                embedding = model(images)
                loss = tf.reduce_sum(embedding * embedding)
            grads = tape.gradient(loss, model.trainable_variables)
            self.assertTrue(np.isfinite(float(loss)))
            for var, grad in zip(model.trainable_variables, grads):
                self.assertIsNotNone(grad, msg=f"{var.name} has no gradient")
                self.assertGreater(float(tf.reduce_max(tf.abs(grad))), 0.0,
                                   msg=f"{var.name} has no gradient at ring16")

    def test_toy_embedding_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy(root)
            images = np.random.default_rng(1).uniform(-1, 1, size=(4, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=6, total_bits=15,
                                 ring_bits=32, dtype=tf.float64)
            emb = model(images).numpy()
            ref = evaluate_numpy(path, images, fixed_point=FP, ring_bits=32)
            np.testing.assert_array_equal(
                np.rint(emb * 64).astype(np.int64),
                np.rint(ref * 64).astype(np.int64))

    def test_separate_kernel_grid_is_bit_exact_against_the_oracle(self):
        # Public kernels on a finer grid than the activations.  The
        # forward and the oracle have to agree there too, fusions included,
        # otherwise the training mirror drifts from the emitted program.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy(root)
            images = np.random.default_rng(11).uniform(-1, 1, size=(3, 2, 2, 3))
            for weight_bits in (6, 9, 12):
                model = RingQATModel(path, fractional_bits=6, total_bits=15,
                                     ring_bits=32, dtype=tf.float64,
                                     weight_fractional_bits=weight_bits)
                _, captured = model(images, capture=True)
                _, oracle = evaluate_numpy(
                    path, images, capture=True, fixed_point=FP, ring_bits=32,
                    weight_fractional_bits=weight_bits)
                got = _tf_ints(captured, 6)
                want = _oracle_ints(oracle, 6)
                for name, target in want.items():
                    np.testing.assert_array_equal(
                        got[name], target,
                        err_msg=f"op {name} diverges at f_w={weight_bits}")

    def test_gradients_flow_through_the_ste(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._toy(root)
            images = np.random.default_rng(2).uniform(-1, 1, size=(4, 2, 2, 3))
            model = RingQATModel(path, fractional_bits=6, total_bits=15,
                                 ring_bits=32, dtype=tf.float32)
            with tf.GradientTape() as tape:
                embedding = model(images)
                loss = tf.reduce_sum(embedding * embedding)
            grads = tape.gradient(loss, model.trainable_variables)
            self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(all(g is not None for g in grads))
            # Every trainable parameter of this toy reaches the loss; the STE
            # must not sever any of them.
            for var, grad in zip(model.trainable_variables, grads):
                self.assertGreater(float(tf.reduce_max(tf.abs(grad))), 0.0,
                                   msg=f"{var.name} has no gradient")

    @unittest.skipUnless(_MODEL_R32.is_file(), "model bundle not fetched")
    def test_real_model_slice_is_bit_exact(self):
        # In-distribution faces stay inside the calibrated bounds, where the
        # ring never wraps and the float64 mirror is bit-exact end to end.
        images = np.random.default_rng(3).uniform(-0.3, 0.3, size=(2, 112, 112, 3))
        model = RingQATModel(_MODEL_R32, fractional_bits=6, total_bits=15,
                             ring_bits=32, dtype=tf.float64)
        embedding, captured = model(images, capture=True)
        reference, oracle = evaluate_numpy(_MODEL_R32, images, capture=True,
                                           fixed_point=FP, ring_bits=32)
        got = _tf_ints(captured, 6)
        want = _oracle_ints(oracle, 6)
        mismatched = [name for name, target in want.items()
                      if name in got and not np.array_equal(got[name], target)]
        self.assertEqual(mismatched, [])
        cosine = (np.sum(embedding.numpy() * reference, axis=1) /
                  (np.linalg.norm(embedding.numpy(), axis=1) *
                   np.linalg.norm(reference, axis=1)))
        self.assertTrue(np.all(cosine >= 0.9999))


if __name__ == "__main__":
    unittest.main()
