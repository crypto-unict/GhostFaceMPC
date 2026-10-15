"""The malicious three-party profile (``ps-rep-ring-party``).

The post-sacrifice share type has the replicated layout, so the program keeps
the bounded rounding kernels over the three-way split; it loses only the
two-summand regrouping, which an actively corrupted P0 could exploit.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from ghostmpc.benchmark.process import prepare_player_data
from ghostmpc.benchmark.runner import _protocol_commands
from ghostmpc.codegen import render, verified_io_default
from ghostmpc.core.model import Manifest, Op, TensorRef
from ghostmpc.core.mpspdz_protocols import get_protocol
from ghostmpc.core.sharing import (
    ReplicaMismatch,
    persistence_header,
    read_persistence,
    reconstruct_replicated,
    replicate_components,
    share_ring_n,
    write_persistence,
)

#: The signatures MP-SPDZ itself writes at ring32 (``file_signature<T>()``),
#: captured from a run of both party binaries.  A file whose signature differs
#: is replaced by an empty one, so a drift here would silently drop the input.
SHARE_SIGNATURE = bytes.fromhex(
    "1400000000000000" + "replicated Z2^32".encode().hex() + "20000000")
CLEAR_SIGNATURE = bytes.fromhex(
    "0900000000000000" + "Z2^32".encode().hex() + "20000000")

RING32 = {"fractional_bits": 6, "total_bits": 15, "kernel_bits": 9}


def _manifest() -> Manifest:
    return Manifest(
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


class MaliciousRegistry(unittest.TestCase):
    def test_the_profile_is_abort_secure_at_40_bits(self):
        spec = get_protocol("ps-rep-ring-party")
        self.assertTrue(spec.malicious)
        self.assertEqual(spec.security, "malicious-abort")
        self.assertEqual(spec.security_parameter, 40)
        self.assertEqual((spec.parties, spec.input_parties), (3, 3))
        self.assertFalse(spec.two_summand_split)

    def test_the_ring_width_and_the_binary_reach_the_parties(self):
        spec = get_protocol("ps-rep-ring-party")
        commands = _protocol_commands(
            Path("/tmp/mp-spdz-malicious-test"), spec, "program", 12345, 32)
        self.assertEqual(len(commands), 3)
        for party, command in enumerate(commands):
            self.assertTrue(command[0].endswith("ps-rep-ring-party.x"))
            self.assertEqual(command[1], str(party))
            self.assertEqual(command[command.index("-R") + 1], "32")


class MaliciousEmission(unittest.TestCase):
    def test_post_sacrifice_keeps_the_three_summand_bounded_truncation(self):
        source = render(_manifest(), protocol="ps-rep-ring-party", **RING32)
        self.assertIn("program.use_split(3)", source)
        # The injection is the stock daBit conversion on prefetched material,
        # so ``unsplit`` is turned off explicitly.
        self.assertIn("program.use_unsplit = 0", source)
        self.assertIn("prefetchprep(REQUESTS)", source)
        self.assertIn("prefetch_sync[thread] = thread", source)
        self.assertNotIn("shrsi2", source)
        self.assertIn("from Compiler.instructions import movs, shrsi", source)
        self.assertIn("value.split_to_n_summands(split_k, 3)", source)
        self.assertIn("trunc_round_nearest_bounded(", source)

    def test_the_semi_honest_program_takes_the_two_summand_kernels(self):
        deployed = render(_manifest(), **RING32)
        self.assertIn("program.use_unsplit = 0", deployed)
        self.assertIn("prefetchdabit(REQUESTS)", deployed)
        self.assertNotIn("prefetch_sync", deployed)
        self.assertIn("from Compiler.instructions import shrsi2", deployed)
        self.assertIn("columns = value.split_to_n_summands(k, 2)", deployed)


class VerifiedInputOutput(unittest.TestCase):
    """ps-rep-ring checks the computation, not what enters or leaves it."""

    def test_it_is_the_malicious_default_only(self):
        self.assertTrue(verified_io_default("ps-rep-ring-party"))
        self.assertFalse(verified_io_default("replicated-ring-party"))

    def test_the_client_sharing_is_loaded_without_any_exchange(self):
        source = render(_manifest(), protocol="ps-rep-ring-party", **RING32)
        self.assertIn("_, shares = sint.read_from_file(position, size=n)", source)
        self.assertIn("input_value = secret_input(request_id * INPUT_SIZE, 16)",
                      source)
        # Nothing is re-shared, compared or opened to admit the input: the
        # honest pair fixes its value and ps-rep-ring checks the rest.
        for construct in ("input_from", "require_zero", "verified_components",
                          "checkreplicated", "reveal()"):
            self.assertNotIn(construct, source)

    def test_the_semi_honest_program_and_the_ablation_have_none_of_it(self):
        for source in (render(_manifest(), **RING32),
                       render(_manifest(), protocol="ps-rep-ring-party",
                              verified_io=False, **RING32)):
            self.assertIn("sint.read_from_file(position, size=n)", source)
            for construct in ("check_point",
                              "write_to_file", "claimed_identity"):
                self.assertNotIn(construct, source)

    def test_every_party_receives_its_pair_of_the_replicated_sharing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_player_data(root, 4, 32, values=[1, 2, 3, 4], requests=2,
                                gallery_size=3, gallery_values=[5, 6, 7],
                                identities=[0, 6])
            persistence = root / "Persistence"
            pairs = [read_persistence(persistence / f"Transactions-P{party}.data")
                     for party in range(3)]
            for party in range(3):
                self.assertEqual(
                    (persistence / f"Transactions-clear-P{party}.data")
                    .read_bytes(),
                    CLEAR_SIGNATURE + (0).to_bytes(4, "little")
                    + (6).to_bytes(4, "little"))
        # Component j is P_j's share[0] and P_{j+1}'s share[1].
        for party in range(3):
            self.assertEqual(pairs[party][0], pairs[(party + 1) % 3][1])

        def block(start, size):
            return [[half[start:start + size] for half in pair]
                    for pair in pairs]

        self.assertEqual(reconstruct_replicated(block(0, 3), 32), [5, 6, 7])
        for request in range(2):
            self.assertEqual(
                reconstruct_replicated(block(3 + 4 * request, 4), 32),
                [1, 2, 3, 4])

    def test_the_signatures_are_the_ones_mp_spdz_writes(self):
        self.assertEqual(persistence_header(32), SHARE_SIGNATURE)
        self.assertEqual(persistence_header(32, clear=True), CLEAR_SIGNATURE)

    def test_a_flipped_delivery_is_rejected(self):
        self.assertEqual(
            reconstruct_replicated(
                replicate_components(share_ring_n([1, 0], 3, 32)), 32),
            [1, 0])
        for party in range(3):
            for slot in range(2):
                pairs = replicate_components(share_ring_n([1], 3, 32))
                pairs[party][slot][0] = (pairs[party][slot][0] + 1) % (1 << 32)
                with self.assertRaises(ReplicaMismatch):
                    reconstruct_replicated(pairs, 32)

    def test_the_answer_is_read_after_the_input_shares(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"Transactions-P{party}.data"
                     for party in range(3)]
            for path, pair in zip(paths, replicate_components(
                    share_ring_n([9, 9, 9], 3, 32))):
                write_persistence(path, pair, 32)
            for path, pair in zip(paths, replicate_components(
                    share_ring_n([1, 0], 3, 32))):
                write_persistence(path, pair, 32, append=True)
            pairs = [read_persistence(path, 32, last=2) for path in paths]
            self.assertEqual(reconstruct_replicated(pairs, 32), [1, 0])
            with self.assertRaises(ValueError):
                read_persistence(paths[0], 32, last=6)

    def test_the_persistence_file_is_read_as_share_pairs(self):
        signature = b"Rep3 Z2^32 type string"
        payload = b"".join(
            a.to_bytes(4, "little") + b.to_bytes(4, "little")
            for a, b in ((1, 2), (0xFFFFFFFF, 7)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Transactions-P0.data"
            path.write_bytes(struct.pack("<Q", len(signature)) + signature
                             + payload)
            self.assertEqual(read_persistence(str(path), 32),
                             [[1, 0xFFFFFFFF], [2, 7]])


if __name__ == "__main__":
    unittest.main()
