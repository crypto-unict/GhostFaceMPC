import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ghostmpc import codegen
from ghostmpc.codegen import fusion, program, render, weight_stream
from ghostmpc.codegen.fusion import (
    _fusion_plan,
    _multiply_fusion,
    _multiply_fusion_plan,
    _prelu_fusion,
)
from ghostmpc.codegen.layers import conv, depthwise
from ghostmpc.codegen.layers.conv import _conv2d_spatial
from ghostmpc.codegen.layers.depthwise import (
    _axis_regions,
    _depthwise_conv2d_packed,
    _depthwise_lanes,
    _depthwise_packed_name,
    _same_padding,
)
from ghostmpc.codegen.weight_stream import (
    _prepare_weights,
    _quantize_public_weights,
    _write_public_weights,
)
from ghostmpc.core.model import Manifest, Op, TensorRef


class _FakeVector:
    """A secret vector: the emitted code reads its raw ring value as ``.v``."""

    def __init__(self, values):
        self.v = values


class _FakeRingValue:
    def __init__(self, value):
        self.value = value

    def expand_to_vector(self, size):
        return np.full(size, self.value, dtype=np.int64)


class _FakeScalar:
    """A secret element: the emitted code broadcasts ``.v`` over a tile."""

    def __init__(self, value):
        self.v = _FakeRingValue(value)


class _FakeAddress:
    """Base address of one fake array plus a constant element offset."""

    def __init__(self, array, offset=0):
        self.array = array
        self.offset = int(offset)

    def __add__(self, other):
        return _FakeAddress(self.array, self.offset + int(other))

    __radd__ = __add__


class _FakeAddressVector:
    def __init__(self, array, indices):
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)

    def read(self):
        if self.indices.min(initial=0) < 0 or \
                self.indices.max(initial=0) >= self.array.data.size:
            raise IndexError("gathered read outside the tensor")
        return self.array.data[self.indices]


class _FakeRegint:
    """``incint``: ``base + step * ((i // repeat) % wrap)`` at lane ``i``."""

    @staticmethod
    def inc(size, base, step=1, repeat=1, wrap=None):
        wrap = size if wrap is None else wrap
        lanes = np.arange(size, dtype=np.int64)
        return _FakeAddressVector(
            base.array,
            base.offset + step * ((lanes // repeat) % wrap))


class _FakeCint:
    @staticmethod
    def load_mem(addresses):
        return addresses.read()


class _FakeSecretArray:
    def delete(self):
        pass

    def __init__(self, size):
        self.data = np.zeros(size, dtype=np.int64)

    @property
    def address(self):
        return _FakeAddress(self)

    def __getitem__(self, index):
        if index < 0 or index >= self.data.size:
            raise IndexError(f"read {index} outside the tensor")
        return _FakeScalar(self.data[index])

    def get_vector(self, base, size):
        if base < 0 or base + size > self.data.size:
            raise IndexError(f"read [{base}, {base + size}) outside the tensor")
        return _FakeVector(self.data[base:base + size])

    def assign_vector(self, values, base):
        if base < 0 or base + len(values) > self.data.size:
            raise IndexError(f"write [{base}, {base + len(values)}) outside the tensor")
        self.data[base:base + len(values)] = values


class _FakeSint:
    Array = staticmethod(_FakeSecretArray)

    @staticmethod
    def concat(parts):
        return np.concatenate(parts)

    @staticmethod
    def load_mem(addresses):
        return addresses.read()

    def __call__(self, value, size=1):
        return np.full(size, value, dtype=np.int64)


class _FakePublicArray:
    def delete(self):
        pass

    def __init__(self, values):
        self.data = np.asarray(values, dtype=np.int64)

    @property
    def address(self):
        return _FakeAddress(self)

    def get_vector(self, base, size):
        if base < 0 or base + size > self.data.size:
            raise IndexError(f"read [{base}, {base + size}) outside the weights")
        return self.data[base:base + size]


def _for_range_multithread(threads, budget, count):
    def decorate(body):
        for index in range(count):
            body(index)
    return decorate


def _reference_depthwise(input_values, kernel, strides, pads, output_shape):
    input_h, input_w, _ = input_values.shape
    kernel_h, kernel_w = kernel.shape[:2]
    output_h, output_w = output_shape[0], output_shape[1]
    result = np.zeros(output_shape, dtype=np.int64)
    for out_y in range(output_h):
        for out_x in range(output_w):
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    in_y = out_y * strides[0] + ky - pads[0]
                    in_x = out_x * strides[1] + kx - pads[1]
                    if 0 <= in_y < input_h and 0 <= in_x < input_w:
                        result[out_y, out_x] += input_values[in_y, in_x] * kernel[ky, kx]
    return result


def _reference_conv(input_values, kernel, strides, pads, output_shape):
    input_h, input_w, _ = input_values.shape
    kernel_h, kernel_w = kernel.shape[:2]
    output_h, output_w = output_shape[0], output_shape[1]
    result = np.zeros(output_shape, dtype=np.int64)
    for out_y in range(output_h):
        for out_x in range(output_w):
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    in_y = out_y * strides[0] + ky - pads[0]
                    in_x = out_x * strides[1] + kx - pads[1]
                    if 0 <= in_y < input_h and 0 <= in_x < input_w:
                        result[out_y, out_x] += \
                            input_values[in_y, in_x] @ kernel[ky, kx]
    return result


class CodegenTests(unittest.TestCase):
    def setUp(self):
        self.manifest = Manifest(
            1, "test", "url", "sum", [2, 2, 3], [4],
            [Op("conv", "Conv2D", ["input"], [2, 2, 4],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME", "filters": 4},
                ["conv:0", "conv:1"]),
             Op("act", "PReLU", ["conv"], [2, 2, 4], {"shared_axes": [1, 2]}, ["act:0"]),
             Op("embedding", "Flatten", ["act"], [2, 2, 4])],
            [TensorRef("conv:0", [1, 1, 3, 4], 0, 12),
             TensorRef("conv:1", [4], 12, 4),
             TensorRef("act:0", [4], 16, 4)], "weights.npy", "weights",
        )

    def test_contains_secret_share_input_and_markers(self):
        source = render(self.manifest)
        # The client's replicated sharing is loaded as it is: no server
        # re-shares a component.
        self.assertIn("_, shares = sint.read_from_file(position, size=n)", source)
        self.assertIn("input_value = secret_input(request_id * INPUT_SIZE, 12)",
                      source)
        self.assertNotIn("input_from", source)
        self.assertNotIn("checkreplicated", source)
        self.assertIn("program.unlimited_compilation()", source)
        self.assertIn("GHOSTMPC_MARKER", source)
        self.assertIn("start_timer(1)", source)
        self.assertIn("start_timer(2)", source)
        self.assertIn("start_timer(3)", source)
        self.assertIn(">= 0", source)
        self.assertIn("public_weights", source)
        self.assertIn("trunc_round_nearest_bounded(v_conv_raw.get_vector(), "
                      "sfix.k + sfix.f, sfix.f, signed=True)", source)
        self.assertNotIn(".reveal()", source)

    def test_bounded_truncation_is_the_deployed_rounding(self):
        source = render(
            self.manifest, fractional_bits=6, total_bits=15, kernel_bits=9)
        self.assertIn("def trunc_floor_bounded", source)
        self.assertIn("value.split_to_n_summands(split_k, 3)", source)
        self.assertIn("RING_BITS - split_k + m", source)
        self.assertIn("shifted = sint(size=value.size)", source)
        # The semi-honest ring replaces the entry points by the two-summand
        # kernel; the three-summand helpers stay for the malicious profile.
        self.assertIn("columns = value.split_to_n_summands(split_k, 2)", source)
        # One call replaces the producer truncation and one replaces the
        # separate PReLU negative-branch truncation (plus the definition).
        self.assertEqual(source.count("trunc_round_nearest_bounded("), 3)
        with self.assertRaisesRegex(ValueError, "nearest rounding"):
            render(self.manifest, fractional_bits=6, total_bits=26)

    def test_liveness_deletes_alias_owner_once_after_last_consumer(self):
        source = render(self.manifest)
        # Flatten aliases the final activation, so neither alias may free the
        # returned embedding. The consumed input/raw producer are reclaimed.
        self.assertNotIn("v_act.delete()", source)
        self.assertNotIn("v_embedding.delete()", source)
        self.assertIn("input_value.delete()", source)
        self.assertEqual(source.count("v_conv.delete()"), 1)

    def test_public_weight_writer_and_oracle_share_one_quantizer(self):
        values = np.array([-1.25, -0.5 / 64, 0.5 / 64, 1.25, 3.125])
        expected = _quantize_public_weights(values, 6, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public-input"
            _write_public_weights(path, values, 6, 32)
            written = np.array([int(line) for line in path.read_text().splitlines()])
        np.testing.assert_array_equal(written, expected)

    def test_measured_timer_absent_by_default(self):
        source = render(self.manifest)
        self.assertIn("WARMUPS = 0", source)
        for timer in ("20000", "20001", "20002"):
            self.assertNotIn(f"start_timer({timer})", source)
            self.assertNotIn(f"stop_timer({timer})", source)

    def test_ring32_candidate_can_raise_total_precision_explicitly(self):
        source = render(self.manifest, fractional_bits=7,
                        total_bits=17)
        self.assertIn("sfix.set_precision(7, 17)", source)

    def test_measured_timer_brackets_post_warmup_requests(self):
        source = render(self.manifest, requests=7, warmups=5, measured_timer=True)
        self.assertIn("WARMUPS = 5", source)
        self.assertIn("@if_(request_id >= WARMUPS)", source)
        self.assertIn("start_timer(20000)", source)
        self.assertIn("stop_timer(20000)", source)
        # The measured region opens before the input is shared and closes after
        # the final layer, so it captures the whole request.
        self.assertLess(source.index("start_timer(20000)"),
                        source.index("input_value = secret_input"))
        self.assertGreater(source.rindex("stop_timer(20000)"),
                           source.rindex("start_timer(1)"))

    def test_weight_load_and_input_share_timers(self):
        source = render(self.manifest, requests=7, warmups=5, measured_timer=True)
        # The weight-load timer wraps the one-time public ingestion, before the
        # request loop and unconditionally.
        self.assertIn("start_timer(20001)\npublic_weights", source)
        self.assertIn("public_input()", source)
        self.assertIn("stop_timer(20001)", source)
        self.assertLess(source.index("stop_timer(20001)"),
                        source.index("@for_range(REQUESTS)"))
        # The input-share timer brackets secret_input, gated on the measured
        # requests like the aggregate timer.
        self.assertIn("start_timer(20002)", source)
        self.assertIn("stop_timer(20002)", source)
        self.assertLess(source.index("start_timer(20002)"),
                        source.index("input_value = secret_input"))
        self.assertLess(source.index("input_value = secret_input"),
                        source.index("stop_timer(20002)"))

    def test_measured_timer_warmups_clamped_below_request_count(self):
        # At least one measured request must remain, so warmups never reaches the
        # request count.
        source = render(self.manifest, requests=7, warmups=10, measured_timer=True)
        self.assertIn("WARMUPS = 6", source)

    def test_bit_injection_defaults_to_offline_dabit(self):
        # Pre-generated daBits: ``-E ring`` turns ``unsplit`` on, so the
        # program turns it off explicitly.
        source = render(self.manifest)
        self.assertIn("program.use_split(3)\nprogram.use_unsplit = 0", source)
        self.assertIn("prefetchdabit(REQUESTS)", source)

    def test_offline_dabit_prefetches_before_the_measured_request(self):
        # The daBit conversion stays MP-SPDZ's, ``unsplit`` is turned
        # off explicitly (``-E ring`` enables it), and the daBits of a request
        # are generated before its MEASURED timer starts, under their own.
        source = render(self.manifest, requests=3, warmups=1,
                        measured_timer=True)
        self.assertIn("program.use_split(3)\nprogram.use_unsplit = 0", source)
        self.assertIn("from Compiler.instructions import prefetchdabit", source)
        loop = source[source.index("def _(request_id):"):]
        self.assertLess(loop.index("start_timer(20008)"),
                        loop.index("prefetchdabit(REQUESTS)"))
        self.assertLess(loop.index("prefetchdabit(REQUESTS)"),
                        loop.index("stop_timer(20008)"))
        self.assertLess(loop.index("stop_timer(20008)"),
                        loop.index("start_timer(20000)"))
        self.assertEqual(source.count("prefetchdabit(REQUESTS)"), 1)
        # Without the measured timer the prefetch stays, untimed.
        bare = render(self.manifest)
        self.assertIn("prefetchdabit(REQUESTS)", bare)
        self.assertNotIn("start_timer(20008)", bare)

    def test_prelu_uses_preexpanded_public_alpha(self):
        source = render(self.manifest)
        self.assertIn(
            "v_conv.get_vector().v * public_weights.get_vector(36, 16)",
            source,
        )
        self.assertIn("trunc_round_nearest_bounded(v_act_negative_raw, "
                      "sfix.k + sfix.f, sfix.f, signed=True)", source)
        self.assertIn("sfix.round_nearest = True", source)
        self.assertNotIn("v_act_alpha = cint.Array", source)
        self.assertNotIn("public_weights[16 + i % 4]", source)

    def test_conv_bias_weight_plan_is_preexpanded(self):
        weights = np.arange(20, dtype=np.float64) / 10
        plan = _prepare_weights(self.manifest, weights)
        offset = plan.offsets["conv:conv_expanded_bias"]
        np.testing.assert_array_equal(
            plan.values[offset:offset + 16],
            np.tile(weights[12:16], 4),
        )

    def test_conv_truncation_adds_bias_without_a_serial_loop(self):
        source = render(self.manifest)
        self.assertIn("+ public_weights.get_vector(", source)
        self.assertNotIn("v_conv_bias = cint.Array", source)
        self.assertNotIn("public_weights[12 + i % 4]", source)

    def test_prelu_weight_plan_expands_alpha_in_nhwc_order(self):
        weights = np.arange(20, dtype=np.float64) / 10
        plan = _prepare_weights(self.manifest, weights)
        offset = plan.offsets["act:prelu_expanded_alpha"]
        np.testing.assert_array_equal(
            plan.values[offset:offset + 16],
            np.tile(weights[16:20], 4),
        )

    def test_channel_concat_interleaves_spatial_positions(self):
        manifest = Manifest(
            2, "test", "url", "sum", [2, 1, 1], [2, 1, 2],
            [Op("concat", "Concatenate", ["left", "right"], [2, 1, 2], {"axis": -1})],
            [], "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertIn("v_concat.assign_vector(", source)
        self.assertIn("input_value.get_vector(position * 1, 1)", source)
        self.assertIn("position * 2 + 0", source)
        self.assertIn("position * 2 + 1", source)
        self.assertNotIn("def _(channel)", source)

    def test_conv_1x1_uses_channel_and_spatial_tiles(self):
        source = render(self.manifest)
        self.assertIn("acc_0 = sint(0, size=4)", source)
        self.assertIn("public_weights.get_vector(0 + channel_base, 4)", source)
        self.assertIn(".expand_to_vector(4)", source)
        self.assertIn("v_conv_raw.assign_vector(acc_0", source)
        self.assertNotIn("MemValue(sint(0))", source)

    def test_liveness_reclaims_layer_local_raw_and_broadcast_arrays(self):
        conv = render(self.manifest)
        self.assertIn("v_conv_raw.delete()", conv)

        manifest = Manifest(
            2, "test", "url", "sum", [2, 2, 3], [2, 2, 3],
            [Op("gap", "GlobalAveragePooling2D", ["input"], [3], {}),
             Op("mul", "Multiply", ["input", "gap"], [2, 2, 3], {})],
            [], "weights.npy", "weights",
        )
        gap = render(manifest)
        self.assertIn("v_gap_sums.delete()", gap)
        self.assertIn("v_mul_broadcast_1.delete()", gap)

    def test_wide_conv_expands_each_input_once_per_position(self):
        output_c = 200
        manifest = Manifest(
            2, "test", "url", "sum", [1, 4, 3], [1, 4, output_c],
            [Op("conv", "Conv2D", ["input"], [1, 4, output_c],
                {"kernel_size": [1, 1], "strides": [1, 1],
                 "padding": "SAME", "filters": output_c}, ["conv:0"])],
            [TensorRef("conv:0", [1, 1, 3, output_c], 0, 3 * output_c)],
            "weights.npy", "weights",
        )
        source = render(manifest)
        # The whole channel dimension is one tile, so each (in_c, position)
        # scalar is expanded once instead of once per 64-channel tile.
        self.assertIn(f"acc_0 = sint(0, size={output_c})", source)
        self.assertIn("channel_base = 0 + (tile % 1) * 200", source)
        self.assertEqual(source.count(".expand_to_vector(200)"), 3 * 4)

    def test_spatial_conv_uses_channel_and_spatial_tiles(self):
        manifest = Manifest(
            2, "test", "url", "sum", [3, 9, 2], [1, 7, 3],
            [Op("conv", "Conv2D", ["input"], [1, 7, 3],
                {"kernel_size": [3, 3], "strides": [1, 1],
                 "padding": "VALID", "filters": 3}, ["conv:0"])],
            [TensorRef("conv:0", [3, 3, 2, 3], 0, 54)],
            "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertIn("acc_0 = sint(0, size=3)", source)
        self.assertIn("acc_6 = sint(0, size=3)", source)
        self.assertIn("public_weights.get_vector(0 + channel_base, 3)", source)
        self.assertIn("public_weights.get_vector(51 + channel_base, 3)", source)
        self.assertIn("input_value[input_base_6 + 41].v.expand_to_vector(3)", source)
        self.assertIn("v_conv_raw.assign_vector(acc_0", source)
        self.assertNotIn("MemValue(sint(0))", source)

    def test_spatial_conv_regions_emit_restricted_taps_without_a_padded_copy(self):
        manifest = Manifest(
            2, "test", "url", "sum", [4, 4, 2], [4, 4, 3],
            [Op("conv", "Conv2D", ["input"], [4, 4, 3],
                {"kernel_size": [3, 3], "strides": [1, 1],
                 "padding": "SAME", "filters": 3}, ["conv:0"])],
            [TensorRef("conv:0", [3, 3, 2, 3], 0, 54)],
            "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertNotIn("v_conv_padded", source)
        self.assertNotIn("assign_all(0)", source)
        # The interior region keeps all nine taps; the top-left corner starts at
        # ky = kx = 1 and so loads six fewer weight vectors per input channel.
        self.assertIn(
            "input_base_0 = ((out_y_0 * 1 - 1) * 4 + out_x_0 * 1 - 1) * 2",
            source,
        )
        self.assertIn("out_y_0 = 0 + position_0 // 1", source)
        self.assertIn("v_conv_raw.assign_vector(acc_0, (out_y_0 * 4 + out_x_0) * 3", source)

    def test_spatial_conv_emitted_code_matches_reference(self):
        # Interpret the generated block on plain integers: a region whose taps
        # or input base are off reads outside the tensor or produces a value
        # mismatch against the zero-padded reference.
        for stride in (1, 2):
            with self.subTest(stride=stride):
                self._check_spatial_conv(stride)

    def _check_spatial_conv(self, stride):
        input_h, input_w, input_c, output_c = 7, 9, 2, 3
        kernel_h = kernel_w = 3
        output_h = -(-input_h // stride)
        output_w = -(-input_w // stride)
        op = Op("conv", "Conv2D", ["input"], [output_h, output_w, output_c],
                {"kernel_size": [kernel_h, kernel_w], "strides": [stride, stride],
                 "padding": "SAME", "filters": output_c}, ["conv:0"])
        _, _, pad_h, pad_w = _same_padding(op, [input_h, input_w, input_c])

        rng = np.random.default_rng(0)
        input_values = rng.integers(-40, 40, size=(input_h, input_w, input_c))
        kernel = rng.integers(-9, 9, size=(kernel_h, kernel_w, input_c, output_c))

        # A tile narrower than the output channels exercises the tail batch.
        with mock.patch.object(conv, "CONV_CHANNEL_TILE", 2):
            code = _conv2d_spatial(
                "raw", "source", input_h=input_h, input_w=input_w,
                input_c=input_c, output_h=output_h, output_w=output_w,
                output_c=output_c, kernel_h=kernel_h, kernel_w=kernel_w,
                stride_h=stride, stride_w=stride, pad_h=pad_h, pad_w=pad_w,
                kernel_offset=0,
            )

        source = _FakeSecretArray(input_values.size)
        source.data[:] = input_values.reshape(-1)
        namespace = {
            "sint": _FakeSint(),
            "source": source,
            "public_weights": _FakePublicArray(kernel.reshape(-1)),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the block under test is the fixture
        actual = namespace["raw"].data.reshape(output_h, output_w, output_c)
        expected = _reference_conv(
            input_values, kernel, (stride, stride), (pad_h, pad_w),
            (output_h, output_w, output_c),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_depthwise_multiplier_one_uses_packed_spatial_lanes(self):
        manifest = Manifest(
            2, "test", "url", "sum", [3, 9, 3], [1, 7, 3],
            [Op("depth", "DepthwiseConv2D", ["input"], [1, 7, 3],
                {"kernel_size": [3, 3], "strides": [1, 1],
                 "padding": "VALID", "depth_multiplier": 1}, ["depth:0"])],
            [TensorRef("depth:0", [3, 3, 3, 1], 0, 27)],
            "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertIn("PUBLIC_WEIGHT_COUNT = 216", source)
        self.assertIn("acc = sint(0, size=21)", source)
        self.assertIn("public_weights.get_vector(27, 21)", source)
        self.assertIn("input_value.get_vector(input_base + 60, 21).v", source)
        self.assertIn("v_depth_raw.assign_vector(acc", source)
        self.assertNotIn("v_depth_padded", source)
        self.assertNotIn("MemValue(sint(0))", source)

    def test_depthwise_same_stride_two_packs_full_tiles_and_keeps_channel_tail(self):
        manifest = Manifest(
            2, "test", "url", "sum", [6, 28, 66], [3, 14, 66],
            [Op("depth", "DepthwiseConv2D", ["input"], [3, 14, 66],
                {"kernel_size": [3, 3], "strides": [2, 2],
                 "padding": "SAME", "depth_multiplier": 1}, ["depth:0"])],
            [TensorRef("depth:0", [3, 3, 66, 1], 0, 594)],
            "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertIn("sint.concat((", source)
        # 13 interior columns fit one lane at full channel width: 13 * 66.
        self.assertIn("acc = sint(0, size=858)", source)
        # The boundary column keeps the per-pixel channel batching: 64 + 2.
        self.assertIn("acc = sint(0, size=64)", source)
        self.assertIn("acc = sint(0, size=2)", source)
        self.assertIn("v_depth_raw.assign_vector(acc", source)
        self.assertNotIn("v_depth_padded", source)

    def test_depthwise_weight_plan_repeats_each_kernel_lane_by_position(self):
        manifest = Manifest(
            2, "test", "url", "sum", [3, 9, 3], [1, 7, 3],
            [Op("depth", "DepthwiseConv2D", ["input"], [1, 7, 3],
                {"kernel_size": [3, 3], "strides": [1, 1],
                 "padding": "VALID", "depth_multiplier": 1}, ["depth:0"])],
            [TensorRef("depth:0", [3, 3, 3, 1], 0, 27)],
            "weights.npy", "weights",
        )
        plan = _prepare_weights(manifest, np.arange(27, dtype=np.float64))
        offset = plan.offsets["depth:depthwise_packed:7"]
        packed = plan.values[offset:offset + 189].reshape(9, 21)
        expected = np.tile(np.arange(27).reshape(9, 3), (1, 7))
        np.testing.assert_array_equal(packed, expected)

    def test_depthwise_packed_emitted_code_matches_reference(self):
        # Interpret the generated block on plain integers. The packed lanes read
        # and write whole NHWC pixels, so any channel/pixel aliasing shows up as
        # a value mismatch rather than as a shape error.
        for channels in (5, 6):
            for stride in (1, 2):
                with self.subTest(channels=channels, stride=stride):
                    self._check_packed_depthwise(channels, stride)

    def _check_packed_depthwise(self, channels, stride):
        input_h, input_w = 7, 9
        kernel_h = kernel_w = 3
        output_h = -(-input_h // stride)
        output_w = -(-input_w // stride)
        op = Op("depth", "DepthwiseConv2D", ["input"], [output_h, output_w, channels],
                {"kernel_size": [kernel_h, kernel_w], "strides": [stride, stride],
                 "padding": "SAME", "depth_multiplier": 1}, ["depth:0"])
        input_shape = [input_h, input_w, channels]
        _, _, pad_h, pad_w = _same_padding(op, input_shape)

        rng = np.random.default_rng(0)
        input_values = rng.integers(-40, 40, size=(input_h, input_w, channels))
        kernel = rng.integers(-9, 9, size=(kernel_h, kernel_w, channels))
        kernel_2d = kernel.reshape(kernel_h * kernel_w, channels)

        with mock.patch.object(conv, "CONV_CHANNEL_TILE", 4), \
                mock.patch.object(depthwise, "DEPTHWISE_LANE_CAP", 4 * channels):
            lanes = _depthwise_lanes(op, input_shape)
            self.assertTrue(lanes, "the packed path must be exercised")
            weights = [kernel_2d.reshape(-1)]
            offsets = {"depth:0": 0}
            cursor = kernel_2d.size
            for lane in lanes:
                offsets[_depthwise_packed_name("depth", lane)] = cursor
                packed = np.tile(kernel_2d, (1, lane))
                weights.append(packed.reshape(-1))
                cursor += packed.size
            code = _depthwise_conv2d_packed(
                "depth", "raw", "source", input_h=input_h, input_w=input_w,
                input_c=channels, output_h=output_h, output_w=output_w,
                output_c=channels, kernel_h=kernel_h, kernel_w=kernel_w,
                stride_h=stride, stride_w=stride, pad_h=pad_h, pad_w=pad_w,
                kernel_offset=0, offsets=offsets,
            )

        source = _FakeSecretArray(input_values.size)
        source.data[:] = input_values.reshape(-1)
        namespace = {
            "sint": _FakeSint(),
            "source": source,
            "public_weights": _FakePublicArray(np.concatenate(weights)),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the block under test is the fixture
        actual = namespace["raw"].data.reshape(output_h, output_w, channels)
        expected = _reference_depthwise(
            input_values, kernel, (stride, stride), (pad_h, pad_w),
            (output_h, output_w, channels),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_depthwise_regions_match_zero_padded_reference_for_stride_one_and_two(self):
        def evaluate(input_values, kernel, stride, pad):
            input_h, input_w, channels = input_values.shape
            kernel_h, kernel_w = kernel.shape[:2]
            output_h = (input_h + 2 * pad[0] - kernel_h) // stride[0] + 1
            output_w = (input_w + 2 * pad[1] - kernel_w) // stride[1] + 1
            result = np.zeros((output_h, output_w, channels), dtype=np.int64)
            for y_start, y_count, ky_start, ky_end in _axis_regions(
                    output_h, input_h, kernel_h, stride[0], pad[0]):
                for x_start, x_count, kx_start, kx_end in _axis_regions(
                        output_w, input_w, kernel_w, stride[1], pad[1]):
                    for out_y in range(y_start, y_start + y_count):
                        for out_x in range(x_start, x_start + x_count):
                            for ky in range(ky_start, ky_end):
                                for kx in range(kx_start, kx_end):
                                    result[out_y, out_x] += (
                                        input_values[
                                            out_y * stride[0] + ky - pad[0],
                                            out_x * stride[1] + kx - pad[1],
                                        ] * kernel[ky, kx]
                                    )
            return result

        input_values = np.arange(6 * 8 * 3, dtype=np.int64).reshape(6, 8, 3) - 50
        kernel = np.arange(3 * 3 * 3, dtype=np.int64).reshape(3, 3, 3) - 10
        for stride, pad in (((1, 1), (1, 1)), ((2, 2), (0, 0))):
            actual = evaluate(input_values, kernel, stride, pad)
            expected = np.zeros_like(actual)
            for out_y in range(actual.shape[0]):
                for out_x in range(actual.shape[1]):
                    for ky in range(3):
                        for kx in range(3):
                            in_y = out_y * stride[0] + ky - pad[0]
                            in_x = out_x * stride[1] + kx - pad[1]
                            if 0 <= in_y < input_values.shape[0] and \
                                    0 <= in_x < input_values.shape[1]:
                                expected[out_y, out_x] += input_values[in_y, in_x] * kernel[ky, kx]
            np.testing.assert_array_equal(actual, expected)

    def test_global_average_pool_reduces_and_truncates_vectors(self):
        manifest = Manifest(
            2, "test", "url", "sum", [2, 2, 3], [3],
            [Op("pool", "GlobalAveragePooling2D", ["input"], [3])],
            [], "weights.npy", "weights",
        )
        source = render(manifest)
        self.assertIn("v_pool_sums.assign_vector", source)
        self.assertIn("input_value.get_vector(position * 3, 3).v", source)
        # The averaging constant is public, so it is carried at f + g bits and
        # removed by one nearest truncation instead of being forced onto the
        # activation grid, where a wide window collapses it.
        self.assertIn("v_pool_scaled = v_pool_sums.get_vector() * 64", source)
        self.assertIn("trunc_round_nearest_bounded(\n        v_pool_scaled, 24, 8, signed=True)))", source)
        self.assertNotIn("MemValue(sint(0))", source)

    def test_pooling_reciprocal_survives_wide_windows_at_low_precision(self):
        # At f=6 the activation grid gives rint(64/784) = 0 and rint(64/49) = 1
        # (-23.4 %): the derived scale has to keep the constant resolved.
        for spatial in (49, 196, 784):
            reciprocal, shift, length = codegen.pooling_reciprocal(
                spatial, 64.0, fractional_bits=6, total_bits=15, ring_bits=32)
            error = abs(reciprocal / (1 << shift) * spatial - 1.0)
            self.assertGreater(reciprocal, 0, f"window {spatial} collapsed")
            self.assertLess(error, 0.01, f"window {spatial} off by {error:.3%}")
            self.assertLessEqual(length, 31, f"window {spatial} overflows ring32")

    def test_separate_kernel_grid_keeps_fusions_and_shifts_by_it(self):
        # Public kernels cost no communication per element, so they may sit on a
        # finer grid than the activations; every declared width then carries
        # ``f + f_w`` and the fused truncation drops the kernel scale and the
        # slope scale.  With bound 16 the accumulator is 4 bits and
        # alpha_bits is 6, so at f_w = 9: sign 4+6+9, length 4+6+9+6+1, shift 9+6.
        manifest = _conv_prelu_manifest()
        source = render(manifest, 6, total_bits=15,
                        kernel_bits=9, overflow_bounds={"conv": 16.0})
        self.assertIn("is_negative(v_act_acc, 19)", source)
        self.assertIn("trunc_round_nearest_bounded(v_act_product, 26, 15", source)
        # At f_w == f the emission is the historical one: 4+6+6 and 4+6+6+6+1.
        plain = render(manifest, 6, total_bits=15,
                       overflow_bounds={"conv": 16.0})
        self.assertIn("is_negative(v_act_acc, 16)", plain)
        self.assertIn("trunc_round_nearest_bounded(v_act_product, 23, 12", plain)

    def test_separate_kernel_grid_states_the_plain_truncation(self):
        # A convolution nobody fuses truncates by the kernel scale alone.
        manifest = Manifest(
            2, "plain", "url", "c", [1, 1, 3], [1, 1, 3],
            [Op("conv", "Conv2D", ["input"], [1, 1, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"])],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
             TensorRef("conv:1", [3], 9, 3)], "weights.npy", "w")
        wide = render(manifest, 6, total_bits=15, kernel_bits=9)
        self.assertIn("trunc_round_nearest_bounded(v_conv_raw.get_vector(), "
                      "sfix.k + 9, 9, signed=True)", wide)
        same = render(manifest, 6, total_bits=15)
        self.assertIn("trunc_round_nearest_bounded(v_conv_raw.get_vector(), "
                      "sfix.k + sfix.f, sfix.f, signed=True)", same)

    def test_linear_frontiers_take_the_bounded_split_by_default(self):
        # The bounded identity applies to every linear frontier.
        manifest = Manifest(
            2, "plain", "url", "c", [1, 1, 3], [1, 1, 3],
            [Op("conv", "Conv2D", ["input"], [1, 1, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"])],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
             TensorRef("conv:1", [3], 9, 3)], "weights.npy", "w")
        promoted = render(manifest, 6, total_bits=15,
                          kernel_bits=9)
        self.assertIn(
            "trunc_round_nearest_bounded(v_conv_raw.get_vector(), "
            "sfix.k + 9, 9, signed=True)", promoted)

    def test_separate_kernel_grid_rescales_only_the_kernels(self):
        manifest = _conv_prelu_manifest()
        weights = np.arange(1, 13, dtype=np.float64) / 32.0
        plain = _prepare_weights(manifest, weights, fractional_bits=6,
                                 ring_bits=32, overflow_bounds={"conv": 16.0})
        wide = _prepare_weights(manifest, weights, fractional_bits=6,
                                ring_bits=32, overflow_bounds={"conv": 16.0},
                                kernel_bits=9)
        kernel = "conv:0"
        start, length = plain.offsets[kernel], plain.lengths[kernel]
        np.testing.assert_allclose(
            wide.values[start:start + length],
            plain.values[start:start + length] * 8.0)
        # One global rint(v * 2^f) writes the stream, so the kernel lands on the
        # 2^9 grid exactly.
        quantised = _quantize_public_weights(wide.values, 6, 32)
        np.testing.assert_array_equal(
            quantised[start:start + length],
            np.rint(weights[:length] * (1 << 9)).astype(np.int64))

    def test_pooling_reciprocal_refuses_a_ring_that_cannot_hold_it(self):
        # ring16/f4 cannot even carry the plain grid product of a 28x28 window
        # over values near 2^9; emitting it would wrap silently.
        with self.assertRaises(OverflowError):
            codegen.pooling_reciprocal(784, 512.0, fractional_bits=4,
                                       total_bits=7, ring_bits=16)
        # The same window is fine once the values are small enough to fit.
        reciprocal, shift, length = codegen.pooling_reciprocal(
            784, 2.0, fractional_bits=4, total_bits=7, ring_bits=16)
        self.assertGreater(reciprocal, 0)
        self.assertLessEqual(length, 16)


class _Vec:
    """A ring vector: ``.v`` reads the same value a secret vector exposes."""

    # NumPy defers to the reflected operators instead of broadcasting over a
    # public vector element by element.
    __array_priority__ = 100

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    @property
    def v(self):
        return self

    def round(self, k, f, nearest=False):
        if not nearest:
            raise AssertionError("the generator only emits nearest rounding")
        return _Vec((self.values + (1 << (f - 1))) >> f)

    def expand_to_vector(self, size):
        return _Vec(np.full(size, int(self.values[0]), dtype=np.int64))

    def _other(self, other):
        return other.values if isinstance(other, _Vec) else np.asarray(other)

    def __add__(self, other):
        return _Vec(self.values + self._other(other))

    __radd__ = __add__

    def __mul__(self, other):
        return _Vec(self.values * self._other(other))

    __rmul__ = __mul__

    def __ge__(self, other):
        return _Vec((self.values >= self._other(other)).astype(np.int64))

    def less_than(self, other, bit_length):
        if self.values.max(initial=0) >= 1 << bit_length or \
                self.values.min(initial=0) < -(1 << bit_length):
            raise AssertionError("the comparison declares too few bits")
        return _Vec((self.values < self._other(other)).astype(np.int64))

    def __sub__(self, other):
        return _Vec(self.values - self._other(other))

    def __rsub__(self, other):
        return _Vec(self._other(other) - self.values)

    def if_else(self, a, b):
        return _Vec(np.where(self.values, self._other(a), self._other(b)))


def _is_negative(value, k):
    """Mirror of the emitted ``is_negative`` helper.

    The emitted default resolves to MP-SPDZ's ``less_than(0, k)`` and the
    two-summand profile to the top bit of the split columns; both are the sign
    of the represented value, which is what the fixture models.
    """
    return value.less_than(0, k)


def _truncate_generic(value, k, m):
    """Mirror of the emitted ``truncate_generic`` helper.

    The fixtures only ever run the deployed nearest realization; the
    probabilistic branch of the emitted helper is an ablation with no
    clear-text mirror, so reaching it here is a bug in the fixture.
    """
    return value.round(k, m, nearest=True)


def _trunc_round_nearest_bounded(value, k, m, signed=True):
    """Mirror of the emitted bounded nearest truncation.

    The kernel splits only the declared low bits, but it computes the same
    integer as MP-SPDZ's nearest ``TruncRing`` of a ``k``-bit value.
    """
    return value.round(k, m, nearest=True)


def _generic_truncations(source: str) -> int:
    """Calls to the generic truncation helper, excluding its own definition."""
    return (source.count("truncate_generic(")
            - source.count("def truncate_generic("))


class _FakeRingArray:
    """Stands in for both ``sint.Array`` and ``sfix.Array``."""
    def delete(self):
        pass


    def __init__(self, size):
        self.data = np.zeros(size, dtype=np.int64)

    @property
    def address(self):
        return _FakeAddress(self)

    def __getitem__(self, index):
        return _Vec(self.data[index:index + 1])

    def get_vector(self, base=0, size=None):
        size = self.data.size if size is None else size
        if base < 0 or base + size > self.data.size:
            raise IndexError(f"read [{base}, {base + size}) outside the tensor")
        return _Vec(self.data[base:base + size])

    def assign(self, values):
        self.data[:] = values.values

    def assign_vector(self, values, base=0):
        if base < 0 or base + values.values.size > self.data.size:
            raise IndexError("write outside the tensor")
        self.data[base:base + values.values.size] = values.values


class _FakeFixed:
    Array = staticmethod(_FakeRingArray)
    k = 31
    f = 8
    round_nearest = True

    @staticmethod
    def _new(value):
        return value


class _FakeRingSint:
    Array = staticmethod(_FakeRingArray)

    @staticmethod
    def concat(parts):
        return _Vec(np.concatenate([part.values for part in parts]))

    @staticmethod
    def load_mem(addresses):
        return _Vec(addresses.read())

    def __call__(self, value, size=1):
        return _Vec(np.full(size, value, dtype=np.int64))


class AddFusionTests(unittest.TestCase):
    """Truncation deferred across ``Add(conv, Concatenate(conv, depthwise))``."""

    FRACTIONAL_BITS = 8

    def setUp(self):
        # conv_a feeds both the depthwise and the concatenation, so it is the
        # one producer of the chain that still needs its own truncation.
        self.manifest = Manifest(
            3, "test", "url", "sum", [2, 2, 4], [2, 2, 4],
            [Op("conv_x", "Conv2D", ["input"], [2, 2, 4],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 4}, ["conv_x:0", "conv_x:1"]),
             Op("conv_a", "Conv2D", ["input"], [2, 2, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 2}, ["conv_a:0", "conv_a:1"]),
             Op("dw_b", "DepthwiseConv2D", ["conv_a"], [2, 2, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "depth_multiplier": 1}, ["dw_b:0", "dw_b:1"]),
             Op("concat", "Concatenate", ["conv_a", "dw_b"], [2, 2, 4],
                {"axis": -1}),
             Op("add", "Add", ["conv_x", "concat"], [2, 2, 4])],
            [TensorRef("conv_x:0", [1, 1, 4, 4], 0, 16),
             TensorRef("conv_x:1", [4], 16, 4),
             TensorRef("conv_a:0", [1, 1, 4, 2], 20, 8),
             TensorRef("conv_a:1", [2], 28, 2),
             TensorRef("dw_b:0", [1, 1, 2, 1], 30, 2),
             TensorRef("dw_b:1", [2], 32, 2)], "weights.npy", "weights",
        )
        rng = np.random.default_rng(3)
        self.weights = rng.uniform(-0.5, 0.5, size=34)

    def _run(self, input_values):
        """Interpret the emitted program on plain integers."""
        plan = _prepare_weights(self.manifest, self.weights)
        source = render(self.manifest, self.FRACTIONAL_BITS,
                        weight_offsets=plan.offsets,
                        weight_count=len(plan.values))
        body = source.split("def _(request_id):\n", 1)[1]
        code = "".join(
            line[4:] for line in body.splitlines(keepends=True)
            if not line.strip().startswith(
                ("marker(", "start_timer", "stop_timer", "print_ln", "#", "prefetchdabit(",
                 "input_value = secret_input"))
        )
        input_value = _FakeRingArray(input_values.size)
        input_value.data[:] = np.rint(
            input_values.reshape(-1) * (1 << self.FRACTIONAL_BITS))
        namespace = {
            "sfix": _FakeFixed(), "sint": _FakeRingSint(),
            "input_value": input_value,
            "public_weights": _FakePublicArray(
                np.rint(plan.values * (1 << self.FRACTIONAL_BITS))),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the program under test is the fixture
        return namespace["v_add"].data

    def test_fusion_plan_defers_only_producers_reaching_the_add(self):
        plan = _fusion_plan(self.manifest)
        self.assertEqual(plan.adds, {"add": ("conv_x", "concat")})
        self.assertEqual(set(plan.raw), {"conv_x", "conv_a", "dw_b", "concat"})
        # conv_a also feeds the depthwise, which needs the scaled value.
        self.assertEqual(set(plan.truncated), {"conv_a"})

    def test_ring32_rejects_a_chain_without_accumulator_headroom(self):
        # e + 2f + 1 must stay inside the ring: at ring64 (f=8) a 20-bit
        # accumulator fits (37 <= 63) and the add fuses; at ring32 it does not
        # (37 > 31), so the plan declines rather than wrapping the sum.
        bounds = {"conv_x": 2.0 ** 20, "conv_a": 2.0 ** 20, "dw_b": 2.0 ** 20}
        fused64 = _fusion_plan(self.manifest, bounds=bounds,
                               fractional_bits=self.FRACTIONAL_BITS, ring_bits=64)
        self.assertEqual(set(fused64.adds), {"add"})
        fused32 = _fusion_plan(self.manifest, bounds=bounds,
                               fractional_bits=self.FRACTIONAL_BITS, ring_bits=32)
        self.assertEqual(fused32.adds, {})
        self.assertEqual(set(fused32.raw), set())

    def test_fused_program_matches_the_truncate_everywhere_program(self):
        rng = np.random.default_rng(7)
        input_values = rng.uniform(-1, 1, size=(2, 2, 4))
        fused = self._run(input_values)
        with mock.patch.object(fusion, "_fusion_plan",
                               return_value=codegen.NO_FUSION):
            separate = self._run(input_values)
        # round(a) + round(b) and round(a + b) differ by at most one ulp.
        np.testing.assert_array_less(np.abs(fused - separate), 2)
        self.assertTrue(np.any(fused != 0))

    def test_fused_bias_sums_the_producer_biases_in_channel_order(self):
        plan = _prepare_weights(self.manifest, self.weights)
        offset = plan.offsets["add:add_fused_bias"]
        expected = self.weights[16:20] + np.concatenate(
            [self.weights[28:30], self.weights[32:34]])
        np.testing.assert_allclose(
            plan.values[offset:offset + 16], np.tile(expected, 4))
        self.assertNotIn("conv_x:conv_expanded_bias", plan.offsets)
        self.assertNotIn("dw_b:conv_expanded_bias", plan.offsets)
        self.assertIn("conv_a:conv_expanded_bias", plan.offsets)

class PreluFusionTests(unittest.TestCase):
    """Truncation of a convolution folded into the PReLU that consumes it."""

    FRACTIONAL_BITS = 8

    def setUp(self):
        self.manifest = Manifest(
            3, "test", "url", "sum", [2, 2, 4], [2, 2, 3],
            [Op("conv", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"]),
             Op("act", "PReLU", ["conv"], [2, 2, 3], {"shared_axes": [1, 2]},
                ["act:0"])],
            [TensorRef("conv:0", [1, 1, 4, 3], 0, 12),
             TensorRef("conv:1", [3], 12, 3),
             TensorRef("act:0", [3], 15, 3)], "weights.npy", "weights",
            overflow_bounds={"conv": 2.0 ** 20},
        )
        rng = np.random.default_rng(5)
        self.weights = np.concatenate([
            rng.uniform(-0.5, 0.5, size=15), rng.uniform(0.05, 0.35, size=3),
        ])

    def _run(self, input_values):
        """Interpret the emitted program on plain integers."""
        plan = _prepare_weights(self.manifest, self.weights,
                                fractional_bits=self.FRACTIONAL_BITS)
        source = render(self.manifest, self.FRACTIONAL_BITS,
                        weight_offsets=plan.offsets,
                        weight_count=len(plan.values))
        body = source.split("def _(request_id):\n", 1)[1]
        code = "".join(
            line[4:] for line in body.splitlines(keepends=True)
            if not line.strip().startswith(
                ("marker(", "start_timer", "stop_timer", "print_ln", "#", "prefetchdabit(",
                 "input_value = secret_input"))
        )
        input_value = _FakeRingArray(input_values.size)
        input_value.data[:] = np.rint(
            input_values.reshape(-1) * (1 << self.FRACTIONAL_BITS))
        fixed = _FakeFixed()
        fixed.f = self.FRACTIONAL_BITS
        namespace = {
            "sfix": fixed, "sint": _FakeRingSint(),
            "input_value": input_value,
            "public_weights": _FakePublicArray(
                np.rint(plan.values * (1 << self.FRACTIONAL_BITS))),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the program under test is the fixture
        return namespace["v_act"].data

    def test_fusion_selects_the_sole_producer_of_the_prelu(self):
        plan = _prelu_fusion(self.manifest, None, self.FRACTIONAL_BITS,
                                     codegen.NO_FUSION)
        self.assertEqual(set(plan), {"act"})
        self.assertEqual(plan["act"].producer, "conv")
        self.assertEqual(plan["act"].accumulator_bits, 20)
        # 62 - 2 * 8 - 20 leaves more headroom than the alpha needs.
        self.assertEqual(plan["act"].alpha_bits, self.FRACTIONAL_BITS)

    def test_ring32_narrows_the_fused_truncation_budget(self):
        # At ring32 the fused truncation width is 30 bits, so f_a = 30 - 2f - e.
        # A tight bound still fuses (with the narrower slope); the wide default
        # bound that fuses at ring64 no longer fits.
        self.manifest.overflow_bounds = {"conv": 2.0 ** 4}
        tight = _prelu_fusion(self.manifest, None, self.FRACTIONAL_BITS,
                                      codegen.NO_FUSION, ring_bits=32)
        self.assertEqual(set(tight), {"act"})
        self.assertEqual(tight["act"].alpha_bits, self.FRACTIONAL_BITS)
        self.manifest.overflow_bounds = {"conv": 2.0 ** 20}
        self.assertEqual(
            _prelu_fusion(self.manifest, None, self.FRACTIONAL_BITS,
                                  codegen.NO_FUSION, ring_bits=32),
            {})

    def test_low_fractional_precision_floors_the_alpha_gate_at_f(self):
        # At f=6 the slope can hold at most 6 bits, so demanding the ring64
        # default of 8 would reject every fusion; the floor drops to f and the
        # tight bound fuses with a 6-bit slope.
        self.manifest.overflow_bounds = {"conv": 2.0 ** 4}
        plan = _prelu_fusion(self.manifest, None, 6,
                                     codegen.NO_FUSION, ring_bits=32)
        self.assertEqual(set(plan), {"act"})
        self.assertEqual(plan["act"].alpha_bits, 6)

    def test_fusion_gate_rejects_an_accumulator_without_ring_headroom(self):
        for bound, reason in ((2.0 ** 40, "too few alpha bits"), (0.0, "no bound")):
            with self.subTest(reason=reason):
                self.manifest.overflow_bounds = {"conv": bound}
                self.assertEqual(
                    _prelu_fusion(self.manifest, None,
                                          self.FRACTIONAL_BITS, codegen.NO_FUSION),
                    {})

    def test_fusion_skips_a_producer_with_another_consumer(self):
        self.manifest.ops.append(
            Op("tail", "Add", ["conv", "act"], [2, 2, 3]))
        self.assertEqual(
            _prelu_fusion(self.manifest, None, self.FRACTIONAL_BITS,
                                  codegen.NO_FUSION),
            {})

    def test_fused_weight_plan_quantizes_alpha_and_prescales_the_bias(self):
        # A tighter fractional scale leaves the alpha fewer bits than the
        # activation, which is what the gate trades against the truncation.
        fractional_bits = 12
        self.manifest.overflow_bounds = {"conv": 2.0 ** 30}
        plan = _prepare_weights(self.manifest, self.weights,
                                fractional_bits=fractional_bits)
        alpha_offset = plan.offsets["act:prelu_expanded_alpha"]
        np.testing.assert_array_equal(
            plan.values[alpha_offset:alpha_offset + 12],
            np.tile(np.rint(self.weights[15:18] * (1 << 8)) / (1 << fractional_bits), 4),
        )
        bias_offset = plan.offsets["conv:conv_raw_bias"]
        np.testing.assert_allclose(
            plan.values[bias_offset:bias_offset + 12],
            np.tile(self.weights[12:15] * (1 << fractional_bits), 4),
        )
        self.assertNotIn("conv:conv_expanded_bias", plan.offsets)

class MultiplyFusionTests(unittest.TestCase):
    """SE ``Multiply`` truncation folded into its 1x1 convolution consumer."""

    FRACTIONAL_BITS = 8

    def setUp(self):
        # feat is a full [2, 2, 3] map; gate is the [1, 1, 3] hard-sigmoid
        # scaling, produced here by a stride-2 1x1 conv.  The SE Multiply feeds
        # one 1x1 stride-1 conv, which is the fusion consumer.
        self.manifest = Manifest(
            3, "test", "url", "sum", [2, 2, 3], [2, 2, 3],
            [Op("feat", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["feat:0", "feat:1"]),
             Op("gate", "Conv2D", ["input"], [1, 1, 3],
                {"kernel_size": [1, 1], "strides": [2, 2], "padding": "SAME",
                 "filters": 3}, ["gate:0", "gate:1"]),
             Op("mul", "Multiply", ["feat", "gate"], [2, 2, 3]),
             Op("conv", "Conv2D", ["mul"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"]),
             Op("embedding", "Flatten", ["conv"], [2, 2, 3])],
            [TensorRef("feat:0", [1, 1, 3, 3], 0, 9),
             TensorRef("feat:1", [3], 9, 3),
             TensorRef("gate:0", [1, 1, 3, 3], 12, 9),
             TensorRef("gate:1", [3], 21, 3),
             TensorRef("conv:0", [1, 1, 3, 3], 24, 9),
             TensorRef("conv:1", [3], 33, 3)], "weights.npy", "weights",
            overflow_bounds={"conv": 2.0 ** 8},
        )
        rng = np.random.default_rng(15)
        self.weights = rng.uniform(-0.4, 0.4, size=36)

    def _plan(self, **kwargs):
        bounds = kwargs.pop("bounds", self.manifest.overflow_bounds)
        return _multiply_fusion_plan(
            self.manifest, bounds, self.FRACTIONAL_BITS, **kwargs)

    def test_plan_selects_the_se_multiply_and_its_conv(self):
        plan = self._plan()
        self.assertEqual(set(plan), {"mul"})
        fused = plan["mul"]
        self.assertEqual(fused.conv, "conv")
        self.assertEqual(fused.feature, "feat")
        self.assertEqual(fused.gate, "gate")
        self.assertEqual(fused.accumulator_bits, 8)

    def _with_add_chain(self):
        # Route the fused conv into an Add alongside another raw conv, so the
        # planner sees the ``2^{3f}`` add chain Rollout B has to certify.
        self.manifest.ops.insert(
            4, Op("add", "Add", ["conv", "feat"], [2, 2, 3]))
        self.manifest.ops[5].inputs = ["add"]
        return codegen.FusionPlan(
            {"add": ("conv", "feat")},
            frozenset({"conv", "feat"}), frozenset())

    def test_plan_fuses_a_conv_whose_add_chain_fits_at_s3(self):
        fusion = self._with_add_chain()
        plan = self._plan(bounds={"conv": 2.0 ** 2, "feat": 2.0 ** 7},
                          fusion=fusion)
        self.assertEqual(set(plan), {"mul"})

    def test_plan_skips_a_conv_whose_add_chain_overflows_at_s3(self):
        fusion = self._with_add_chain()
        # The conv's own accumulator fits ring32 (2 + 3f = 26 <= 30), but the
        # widest chain bound does not (7 + 3f + 1 = 32 > 31), so the pair falls
        # back to the current path.
        self.assertEqual(
            self._plan(bounds={"conv": 2.0 ** 2, "feat": 2.0 ** 7},
                       fusion=fusion, ring_bits=32), {})

    def test_plan_skips_a_multiply_with_more_than_one_consumer(self):
        self.manifest.ops.append(Op("tail", "Add", ["mul", "conv"], [2, 2, 3]))
        self.assertEqual(self._plan(), {})

    def test_plan_skips_a_non_1x1_or_strided_conv(self):
        self.manifest.ops[3].params["kernel_size"] = [3, 3]
        self.assertEqual(self._plan(), {})
        self.manifest.ops[3].params["kernel_size"] = [1, 1]
        self.manifest.ops[3].params["strides"] = [2, 2]
        self.assertEqual(self._plan(), {})

    def test_plan_needs_a_bound_that_fits_the_ring(self):
        self.assertEqual(self._plan(bounds={}), {})
        # e + 3f must leave the ring two bits: 2^54 needs 54 + 24 > 62.
        self.assertEqual(self._plan(bounds={"conv": 2.0 ** 54}), {})

    def _run(self, input_values, *, fuse):
        plan = _prepare_weights(self.manifest, self.weights,
                                fractional_bits=self.FRACTIONAL_BITS)
        source = render(self.manifest, self.FRACTIONAL_BITS,
                        weight_offsets=plan.offsets,
                        weight_count=len(plan.values))
        body = source.split("def _(request_id):\n", 1)[1]
        code = "".join(
            line[4:] for line in body.splitlines(keepends=True)
            if not line.strip().startswith(
                ("marker(", "start_timer", "stop_timer", "print_ln", "#", "prefetchdabit(",
                 "input_value = secret_input"))
        )
        input_value = _FakeRingArray(input_values.size)
        input_value.data[:] = np.rint(
            input_values.reshape(-1) * (1 << self.FRACTIONAL_BITS))
        fixed = _FakeFixed()
        fixed.f = self.FRACTIONAL_BITS
        namespace = {
            "sfix": fixed, "sint": _FakeRingSint(),
            "input_value": input_value,
            "public_weights": _FakePublicArray(
                np.rint(plan.values * (1 << self.FRACTIONAL_BITS))),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the program under test is the fixture
        return namespace["v_conv"].data

    def _integer_oracle(self, input_values):
        f = self.FRACTIONAL_BITS
        scale = 1 << f
        half, half2 = 1 << (f - 1), 1 << (2 * f - 1)
        w = np.rint(self.weights * scale).astype(np.int64)
        xin = np.rint(input_values * scale).astype(np.int64)
        fw, fbias = w[0:9].reshape(3, 3), w[9:12]
        gw, gbias = w[12:21].reshape(3, 3), w[21:24]
        cw, cbias = w[24:33].reshape(3, 3), w[33:36]
        feat = ((xin.reshape(4, 3) @ fw + half) >> f) + fbias
        gate = ((xin[0, 0] @ gw + half) >> f) + gbias
        raw = feat * gate  # [1, 1, 3] gate broadcast over the four positions
        conv = ((raw @ cw + half2) >> (2 * f)) + cbias
        return conv.reshape(-1)

    def test_batched_reference_keeps_fused_se_accumulation_exact(self):
        import tempfile

        from ghostmpc.eval import reference

        images = np.random.default_rng(31).uniform(-1, 1, size=(3, 2, 2, 3))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            self.manifest.save(manifest_path)
            np.save(Path(directory) / "weights.npy", self.weights)
            batched = reference.evaluate_numpy(
                manifest_path, images,
                fixed_point=(self.FRACTIONAL_BITS, 31))
            singles = np.stack([
                reference.evaluate_numpy(
                    manifest_path, image,
                    fixed_point=(self.FRACTIONAL_BITS, 31))
                for image in images
            ])
        np.testing.assert_array_equal(batched, singles)


class MultiplySeAddChainFusionTests(unittest.TestCase):
    """Rollout B: a fused SE conv that also defers into an ``Add`` at ``2^{3f}``.

    ``cf`` is fed by the SE ``Multiply`` (so its accumulator sits at ``2^{3f}``)
    and feeds both a depthwise (which needs the truncated ``2^f`` value) and the
    ``Concatenate`` that reaches the fused ``Add`` (which needs the raw ``2^{3f}``
    accumulator).  The chain is the shape of ``conv2d_11 -> concatenate_7 ->
    add_3`` in the real model.
    """

    FRACTIONAL_BITS = 8

    def setUp(self):
        self.manifest = Manifest(
            3, "test", "url", "sum", [2, 2, 2], [2, 2, 4],
            [Op("feat", "Conv2D", ["input"], [2, 2, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 2}, ["feat:0", "feat:1"]),
             Op("gate", "Conv2D", ["input"], [1, 1, 2],
                {"kernel_size": [1, 1], "strides": [2, 2], "padding": "SAME",
                 "filters": 2}, ["gate:0", "gate:1"]),
             Op("mul", "Multiply", ["feat", "gate"], [2, 2, 2]),
             Op("cf", "Conv2D", ["mul"], [2, 2, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 2}, ["cf:0", "cf:1"]),
             Op("dw", "DepthwiseConv2D", ["cf"], [2, 2, 2],
                {"kernel_size": [3, 3], "strides": [1, 1], "padding": "SAME",
                 "depth_multiplier": 1}, ["dw:0", "dw:1"]),
             Op("cat", "Concatenate", ["cf", "dw"], [2, 2, 4], {"axis": -1}),
             Op("side", "Conv2D", ["input"], [2, 2, 4],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 4}, ["side:0", "side:1"]),
             Op("add", "Add", ["side", "cat"], [2, 2, 4]),
             Op("embedding", "Flatten", ["add"], [2, 2, 4])],
            [TensorRef("feat:0", [1, 1, 2, 2], 0, 4),
             TensorRef("feat:1", [2], 4, 2),
             TensorRef("gate:0", [1, 1, 2, 2], 6, 4),
             TensorRef("gate:1", [2], 10, 2),
             TensorRef("cf:0", [1, 1, 2, 2], 12, 4),
             TensorRef("cf:1", [2], 16, 2),
             TensorRef("dw:0", [3, 3, 2, 1], 18, 18),
             TensorRef("dw:1", [2], 36, 2),
             TensorRef("side:0", [1, 1, 2, 4], 38, 8),
             TensorRef("side:1", [4], 46, 4)],
            "weights.npy", "weights",
            overflow_bounds={"cf": 2.0 ** 10, "dw": 2.0 ** 10,
                             "side": 2.0 ** 10},
        )
        rng = np.random.default_rng(7)
        self.weights = rng.uniform(-0.3, 0.3, size=50)

    def _source(self, *, fuse):
        plan = _prepare_weights(self.manifest, self.weights,
                                fractional_bits=self.FRACTIONAL_BITS)
        source = render(self.manifest, self.FRACTIONAL_BITS,
                        weight_offsets=plan.offsets,
                        weight_count=len(plan.values))
        return source, plan

    def _run(self, input_values, *, fuse):
        source, plan = self._source(fuse=fuse)
        body = source.split("def _(request_id):\n", 1)[1]
        code = "".join(
            line[4:] for line in body.splitlines(keepends=True)
            if not line.strip().startswith(
                ("marker(", "start_timer", "stop_timer", "print_ln", "#", "prefetchdabit(",
                 "input_value = secret_input")))
        input_value = _FakeRingArray(input_values.size)
        input_value.data[:] = np.rint(
            input_values.reshape(-1) * (1 << self.FRACTIONAL_BITS))
        fixed = _FakeFixed()
        fixed.f = self.FRACTIONAL_BITS
        namespace = {
            "sfix": fixed, "sint": _FakeRingSint(),
            "input_value": input_value,
            "public_weights": _FakePublicArray(
                np.rint(plan.values * (1 << self.FRACTIONAL_BITS))),
            "for_range_multithread": _for_range_multithread,
            "truncate_generic": _truncate_generic,
            "trunc_round_nearest_bounded": _trunc_round_nearest_bounded,
            "is_negative": _is_negative,
            "for_range": lambda count: (lambda body: [body(i) for i in range(count)]),
            "regint": _FakeRegint, "cint": _FakeCint,
            "THREADS": 1,
        }
        exec(code, namespace)  # noqa: S102 - the program under test is the fixture
        return namespace["v_add"].data

    def test_planner_fuses_the_conv_that_also_defers_into_the_add(self):
        fusion = _fusion_plan(self.manifest,
                              bounds=self.manifest.overflow_bounds,
                              fractional_bits=self.FRACTIONAL_BITS)
        plan = _multiply_fusion(
            self.manifest, self.manifest.overflow_bounds,
            self.FRACTIONAL_BITS, fusion)
        self.assertEqual({p.conv for p in plan.values()}, {"cf"})
        self.assertIn("cf", fusion.raw)
        self.assertIn("cf", fusion.truncated)

    def test_fused_chain_matches_the_baseline_within_one_rounding(self):
        from ghostmpc.eval import reference

        rng = np.random.default_rng(37)
        input_values = rng.uniform(-1, 1, size=(2, 2, 2))
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            self.manifest.save(manifest_path)
            np.save(Path(directory) / "weights.npy", self.weights)
            kwargs = dict(fixed_point=(self.FRACTIONAL_BITS, 31),
                          overflow_bounds=self.manifest.overflow_bounds)
            fused = reference.evaluate_numpy(
                manifest_path, input_values, **kwargs)
            separate = reference.evaluate_numpy(
                manifest_path, input_values, **kwargs)
        # Deferring the Multiply truncation into the shared 2f shift moves each
        # output by at most the one rounding the two paths no longer share.
        np.testing.assert_array_less(
            np.abs(fused - separate) * (1 << self.FRACTIONAL_BITS), 3)


if __name__ == "__main__":
    unittest.main()


def _conv_prelu_manifest() -> Manifest:
    """conv -> fused PReLU, the smallest graph carrying a product scale."""
    return Manifest(
        2, "kernelgrid", "url", "c", [1, 1, 3], [1, 1, 3],
        [Op("conv", "Conv2D", ["input"], [1, 1, 3],
            {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
             "filters": 3}, ["conv:0", "conv:1"]),
         Op("act", "PReLU", ["conv"], [1, 1, 3], {"shared_axes": [1, 2]},
            ["act:0"])],
        [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
         TensorRef("conv:1", [3], 9, 3),
         TensorRef("act:0", [3], 12, 3)], "weights.npy", "w",
        overflow_bounds={"conv": 16.0})
