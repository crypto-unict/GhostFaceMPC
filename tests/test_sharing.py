import tempfile
import unittest
from pathlib import Path

from ghostmpc.core.sharing import (
    RING,
    decode_fixed,
    encode_fixed,
    persistence_header,
    read_persistence,
    reconstruct_replicated,
    reconstruct_ring_n,
    replicate_components,
    share_fixed_n,
    share_ring_n,
    write_persistence,
)


class SharingTests(unittest.TestCase):
    def test_round_trip_fixed_point(self):
        values = [-1.0, -0.125, 0.0, 0.333, 1.0]
        shares = share_fixed_n(values, 20, 3)
        reconstructed = decode_fixed(reconstruct_ring_n(shares), 20)
        for actual, expected in zip(reconstructed, values):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_round_trip_fixed_point_ring32(self):
        values = [-1.0, -0.125, 0.0, 0.333, 1.0]
        shares = share_fixed_n(values, 12, 3, ring_bits=32)
        reconstructed = decode_fixed(
            reconstruct_ring_n(shares, ring_bits=32), 12, ring_bits=32)
        for actual, expected in zip(reconstructed, values):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_encode_rejects_value_outside_ring32(self):
        # 2^20 * 2^12 = 2^32 wraps the signed 32-bit half-ring; ring64 holds it.
        encode_fixed([1 << 20], 12, ring_bits=64)
        with self.assertRaises(OverflowError):
            encode_fixed([1 << 20], 12, ring_bits=32)

    def test_fresh_shares(self):
        values = [7, 8, 9]
        first = share_ring_n(values, 3)
        second = share_ring_n(values, 3)
        self.assertNotEqual(first[0], second[0])

    def test_ring_wrap(self):
        shares = share_ring_n([RING - 1], 3)
        self.assertEqual(sum(part[0] for part in shares) % RING, RING - 1)

    def test_three_party_components(self):
        shares = share_ring_n([1, 2, RING - 1], 3)
        self.assertEqual(reconstruct_ring_n(shares), [1, 2, RING - 1])


class ClientReplicatedSharingTests(unittest.TestCase):
    """The client writes the replicated sharing the servers compute on."""

    def test_each_component_reaches_exactly_its_two_holders(self):
        pairs = replicate_components(share_ring_n([5, 6], 3))
        for party in range(3):
            self.assertEqual(pairs[party][0], pairs[(party + 1) % 3][1])
        self.assertEqual(reconstruct_replicated(pairs), [5, 6])

    def test_the_share_file_round_trips_at_both_rings(self):
        for ring_bits in (32, 64):
            with self.subTest(ring_bits=ring_bits), \
                    tempfile.TemporaryDirectory() as directory:
                values = [0, 1, (1 << ring_bits) - 1]
                paths = [Path(directory) / f"Transactions-P{party}.data"
                         for party in range(3)]
                for path, pair in zip(paths, replicate_components(
                        share_ring_n(values, 3, ring_bits))):
                    write_persistence(path, pair, ring_bits)
                    self.assertTrue(path.read_bytes().startswith(
                        persistence_header(ring_bits)))
                    self.assertEqual(
                        len(path.read_bytes()),
                        len(persistence_header(ring_bits))
                        + 3 * 2 * ring_bits // 8)
                pairs = [read_persistence(path, ring_bits) for path in paths]
                self.assertEqual(reconstruct_replicated(pairs, ring_bits),
                                 values)

    def test_appending_keeps_one_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Transactions-P0.data"
            write_persistence(path, [[1], [2]])
            write_persistence(path, [[3, 4], [5, 6]], append=True)
            self.assertEqual(read_persistence(path), [[1, 3, 4], [2, 5, 6]])
            write_persistence(path, [[7], [8]])
            self.assertEqual(read_persistence(path), [[7], [8]])


if __name__ == "__main__":
    unittest.main()
