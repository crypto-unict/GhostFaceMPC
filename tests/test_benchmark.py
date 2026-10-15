import struct
import sys
import tempfile
import unittest
from pathlib import Path

from ghostmpc.benchmark import (
    BenchmarkTimeout,
    Trial,
    parse_request_timer,
    prepare_player_data,
    run_commands,
    summarize,
    write_report,
)
from ghostmpc.core.model import DEPLOYMENT_RING_BITS, Manifest
from ghostmpc.core.sharing import read_persistence, reconstruct_replicated


class BenchmarkTests(unittest.TestCase):
    def test_summary_and_report(self):
        trials = [
            Trial("conv", "online", 2, 1, 1.2, 10, 11, 3),
            Trial("conv", "online", 4, 2, 2.2, 20, 21, 4),
        ]
        row = summarize(trials)[0]
        self.assertEqual(row["wall_seconds"]["median"], 3)
        manifest = Manifest(1, "test", "url", "c", [1], [1], [], [], "w", "w")
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = write_report(trials, manifest, Path(directory), "commit")
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

    def test_prepare_player_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_player_data(root, 4, values=[1, 2, 3, 4])
            pairs = [read_persistence(
                root / "Persistence" / f"Transactions-P{party}.data")
                for party in range(3)]
            self.assertEqual(reconstruct_replicated(pairs), [1, 2, 3, 4])
            self.assertFalse((root / "Player-Data").exists())
            self.assertFalse(
                (root / "Persistence" / "Transactions-clear-P0.data").exists())

    def test_the_naive_layout_keeps_one_additive_component_per_party(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_player_data(root, 4, layout="input-binary")
            self.assertTrue((root / "Player-Data").is_dir())
            self.assertFalse((root / "Persistence").exists())
            def read(path):
                raw = path.read_bytes()
                return [x[0] % (1 << DEPLOYMENT_RING_BITS)
                        for x in struct.iter_unpack("<q", raw)]
            p0 = read(root / "Player-Data" / "Input-Binary-P0-0")
            p1 = read(root / "Player-Data" / "Input-Binary-P1-0")
            p2 = read(root / "Player-Data" / "Input-Binary-P2-0")
            self.assertEqual(len(p0), 4)
            self.assertTrue(all(
                (a + b + c) % (1 << DEPLOYMENT_RING_BITS) == 0
                for a, b, c in zip(p0, p1, p2)
            ))
            with self.assertRaises(ValueError):
                prepare_player_data(root, 4, layout="input-binary",
                                    identities=[0])

    def test_parse_request_timer_uses_max_wall_sum_traffic_and_max_rounds(self):
        stderr = [
            f"Time10000 = {2 + party / 10} seconds "
            f"({1 + party / 10} MB, {7 + party} rounds, {0.5 + party / 10} CPU seconds)"
            for party in range(3)
        ]
        row = parse_request_timer(stderr)
        self.assertEqual(row["wall_ms"], 2200)
        self.assertEqual(row["bytes_sent_total"], 3_300_000)
        self.assertEqual(row["rounds"], 9)
        self.assertAlmostEqual(row["cpu_ms_total"], 1800)

    def test_parse_request_timer_preserves_online_offline_invariants(self):
        stderr = []
        for party in range(3):
            stderr.append(
                f"Time10000 = {3 + party / 10} seconds "
                f"({3 + party} MB, {9 + party} rounds, {1.5 + party / 10} CPU seconds)\n"
                f"Time10000[online] = {2 + party / 10} seconds "
                f"({2 + party} MB, {6 + party} rounds, {1 + party / 10} CPU seconds)\n"
                "Time10000[offline] = 1 seconds "
                "(1 MB, 3 rounds, 0.5 CPU seconds)")
        phases = parse_request_timer(stderr, require_phases=True)["phases"]
        for metric in ("cpu_seconds_total", "bytes_total"):
            self.assertAlmostEqual(
                phases["total"][metric],
                phases["online"][metric] + phases["offline"][metric])
        self.assertEqual(
            phases["total"]["rounds"],
            phases["online"]["rounds"] + phases["offline"]["rounds"])
        self.assertAlmostEqual(
            phases["total"]["wall_seconds"],
            phases["online"]["wall_seconds"]
            + phases["offline"]["wall_seconds"])

    def test_parse_request_timer_prefers_exact_byte_counters(self):
        stderr = [
            "Time10000 = 3 seconds (53.6463 MB, 9 rounds, 1.5 CPU seconds)\n"
            "Time10000[bytes] = 53646321\n"
            "Time10000[online] = 2 seconds "
            "(40.0001 MB, 6 rounds, 1 CPU seconds)\n"
            "Time10000[online][bytes] = 40000123\n"
            "Time10000[offline] = 1 seconds "
            "(13.6462 MB, 3 rounds, 0.5 CPU seconds)\n"
            "Time10000[offline][bytes] = 13646198"
            for _ in range(3)
        ]
        phases = parse_request_timer(stderr, require_phases=True)["phases"]
        self.assertEqual(phases["total"]["bytes_by_party"], [53_646_321] * 3)
        self.assertEqual(phases["online"]["bytes_by_party"], [40_000_123] * 3)
        self.assertEqual(phases["offline"]["bytes_by_party"], [13_646_198] * 3)

    def test_parse_request_timer_rejects_inconsistent_phase_bytes(self):
        stderr = [
            "Time10000 = 3 seconds (3 MB, 9 rounds, 1.5 CPU seconds)\n"
            "Time10000[online] = 2 seconds (2 MB, 6 rounds, 1 CPU seconds)\n"
            "Time10000[offline] = 1 seconds "
            "(0.5 MB, 3 rounds, 0.5 CPU seconds)"
            for _ in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "total bytes"):
            parse_request_timer(stderr, require_phases=True)

    def test_run_commands_timeout_terminates_only_its_children(self):
        command = [
            sys.executable, "-c",
            "import time; time.sleep(10)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BenchmarkTimeout):
                run_commands(
                    [command, command], Path(directory),
                    timeout_seconds=0.05)


if __name__ == "__main__":
    unittest.main()
