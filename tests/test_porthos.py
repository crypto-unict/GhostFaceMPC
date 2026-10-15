import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ghostmpc.backends.porthos import (
    load_pool_samples,
    lower_onnx_for_porthos,
    reconstruct_additive_output,
    share_additive_input,
    summarize_offline,
    summarize_phase,
    summarize_samples,
)
from ghostmpc.cli import build_parser

REPO = Path(__file__).resolve().parents[1]


def test_cli_defaults_export_the_ring64_manifest():
    exported = build_parser().parse_args(["export-porthos"])
    assert exported.seed == 0
    assert exported.destination == "artifacts/porthos"
    assert exported.manifest == "artifacts/model-ring64-f16-b0/manifest.json"
    assert exported.gallery_rows == 1024 and exported.gallery_row == 0


def test_additive_input_shares_reconstruct_on_the_ring64_grid():
    values = np.array([-1.5, 0.0, 0.25, 3.75])
    left, right = share_additive_input(values, fractional_bits=16, seed=3)
    assert left.dtype == right.dtype == np.uint64
    restored = reconstruct_additive_output(left, right)
    np.testing.assert_array_equal(restored, np.rint(values * (1 << 16)).astype(np.int64))


def _pool(tmp_path: Path) -> Path:
    path = tmp_path / "pool-0.json"
    request = {
        "pool": 0, "wall_seconds": 2.0, "cpu_seconds_total": 3.0,
        "peak_rss_bytes": 10, "bytes_total": 60, "rounds": 5,
        "rounds_by_party": [4, 5, 3], "causal_rounds_validated": True,
        "parties": [{"party": 0, "bytes_sent": 10, "bytes_received": 30},
                    {"party": 1, "bytes_sent": 20, "bytes_received": 20},
                    {"party": 2, "bytes_sent": 30, "bytes_received": 10}],
        "phases": {"online": {"wall_seconds": 2.0, "cpu_seconds_total": 3.0,
                              "bytes_total": 60, "rounds": 5},
                   "sync_barrier": {"wall_seconds": 0.1, "cpu_seconds_total": 0.1,
                                    "bytes_total": 12_000_000, "rounds": 1}},
    }
    path.write_text(json.dumps({
        "backend": "CrypTFlow/Porthos-public-nearest",
        "offline": {"wall_seconds": 9.0, "cpu_seconds_total": 6.0,
                    "bytes_total": 6_000, "rounds": 2,
                    "items": ["session keys", "public weight stream"]},
        "samples": [dict(request, request=0, warmup=True),
                    dict(request, request=1), dict(request, request=2)],
    }))
    return path


def test_the_request_row_excludes_warmups_and_the_barrier(tmp_path: Path):
    samples = load_pool_samples(_pool(tmp_path))
    assert len(samples) == 2
    assert all(row["bytes_reconciled"] for row in samples)
    summary = summarize_samples(samples)
    assert summary["bytes_total"]["median"] == 60
    assert summary["rounds"]["median"] == 5
    assert summarize_phase(samples, "online")["bytes_total"]["median"] == 60
    assert summarize_phase(samples, "sync_barrier")["bytes_total"]["median"] == 12_000_000
    assert summarize_phase(samples, "absent") is None


def test_the_offline_phase_is_one_observation_per_pool(tmp_path: Path):
    offline = summarize_offline(load_pool_samples(_pool(tmp_path)))
    assert offline["observations"] == 1
    assert offline["scope"] == "per pool process"
    assert offline["bytes_total"]["median"] == 6_000
    assert offline["rounds"]["median"] == 2
    assert "session keys" in offline["items"]


def test_prelu_batchnorm_and_attribute_driven_hardsigmoid_lowering(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    tensors = [numpy_helper.from_array(
        np.array([0.25, 0.5], dtype=np.float32).reshape(1, 2, 1, 1), "slope")]
    for name, value in (("gamma", [1, 1]), ("beta", [0, 0]),
                        ("mean", [0, 0]), ("variance", [1, 1])):
        tensors.append(numpy_helper.from_array(np.array(value, np.float32), name))
    graph = helper.make_graph([
        helper.make_node("PRelu", ["input", "slope"], ["prelu"], name="prelu"),
        helper.make_node("BatchNormalization",
                         ["prelu", "gamma", "beta", "mean", "variance"],
                         ["bn"], name="bn", epsilon=1e-5),
        helper.make_node("HardSigmoid", ["bn"], ["output"], name="hs",
                         alpha=0.25, beta=0.25),
    ], "test", [helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
        tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    source, target = tmp_path / "source.onnx", tmp_path / "porthos.onnx"
    onnx.save(model, source)
    result = lower_onnx_for_porthos(source, target)
    assert result["prelu_lowered"] == result["batchnorm_lowered"] == 1
    assert result["hard_sigmoid_lowered"] == 1
    lowered = onnx.load(target)
    arrays = {item.name: numpy_helper.to_array(item)
              for item in lowered.graph.initializer}
    # alpha=.25,beta=.25 gives thresholds -1 and 3; proves attrs are used.
    np.testing.assert_allclose(arrays["hs__lower"], [-1.0])
    np.testing.assert_allclose(arrays["hs__upper"], [3.0])
    assert all(item.doc_string == "MPC_PUBLIC_MODEL_WEIGHT"
               for item in lowered.graph.initializer)


def test_generated_cpp_postprocessor_moves_input_into_persistent_loop():
    module_path = REPO / "baselines" / "porthos" / "postprocess_generated.py"
    spec = importlib.util.spec_from_file_location("porthos_postprocess", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    source = '''#include "ezpc.h"
parseInputs(argc, argv);
vector < int64_t > weight = make_vector<int64_t>( (int32_t)1);
/* Variable to read the clear value corresponding to the input variable weight at (1,1) */
uint64_t __tmp_in_weight;
cin >> __tmp_in_weight;
weight[0] = __tmp_in_weight;
vector < uint64_t > tmp0 = make_vector<uint64_t>( (int32_t)1);
/* Variable to read the clear value corresponding to the input variable tmp0 at (1,1) */
uint64_t __tmp_in_tmp0;
for (uint32_t i0 = 0; i0 < 1; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp0;
}
tmp0[i0] = (role == CLIENT) ? __tmp_in_tmp0 : 0;
}
StartComputation();
vector < uint64_t > output = make_vector<uint64_t>( (int32_t)512);
ClearMemPublic1( (int32_t)1, weight);
EndComputation();
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
cout << (funcReconstruct2PCCons(output[i0], 2)) << endl;
}
cout << "----------------------------------" << endl;
'''
    updated, result = module.transform(source)
    assert result["output_elements"] == 512
    assert "ghostmpc_porthos::request_count()" in updated
    assert "ghostmpc_porthos::write_share" in updated
    assert "funcReconstruct2PCCons" not in updated
    assert "if (PRIMARY) { cin >> __tmp_in_weight; }" in updated
    assert result["retained_public_initializers"] == 1
    assert "retained public initializer weight" in updated
