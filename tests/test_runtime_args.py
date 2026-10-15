"""The party argv the harness launches, and the guards around compilation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from test_malicious import _manifest

from ghostmpc.benchmark.report import write_report
from ghostmpc.benchmark.runner import _protocol_commands
from ghostmpc.benchmark.stats import Trial
from ghostmpc.core.mpspdz_protocols import get_protocol

MPSPDZ = Path("/tmp/mp-spdz-runtime-test")


def _trial(protocol: str) -> Trial:
    return Trial(
        "end_to_end", "total", wall_seconds=1.0, cpu_seconds_p0=1.0,
        cpu_seconds_p1=1.0, bytes_p0_to_p1=1, bytes_p1_to_p0=1, rounds=1,
        protocol=protocol, pool_id="pool", requests=1)


class DeployedRuntimeArguments(unittest.TestCase):
    def test_the_malicious_profile_carries_the_large_preprocessing_batch(self):
        spec = get_protocol("ps-rep-ring-party")
        for command in _protocol_commands(MPSPDZ, spec, "program", 12345, 32):
            self.assertEqual(command[command.index("-b") + 1], "1000000")

    def test_the_semi_honest_argv_carries_no_preprocessing_batch(self):
        spec = get_protocol("replicated-ring-party")
        for command in _protocol_commands(MPSPDZ, spec, "program", 12345, 32):
            self.assertNotIn("-b", command)
            self.assertEqual(command[-1], "-v")

    def test_parties_find_each_other_through_localhost_without_a_network(self):
        spec = get_protocol("replicated-ring-party")
        for command in _protocol_commands(MPSPDZ, spec, "program", 12345, 32):
            self.assertEqual(command[command.index("-h") + 1], "localhost")
            self.assertEqual(command[command.index("-pn") + 1], "12345")


class CompilationReuseGuard(unittest.TestCase):
    """``--skip-compile`` must refuse a program it did not compile.

    A violated reuse assumption does not fail at run time: the run succeeds and
    reports numbers for the previously compiled program, normalized by the
    caller's parameters, so the condition is checked rather than documented.
    """

    def test_a_changed_source_is_refused(self):
        from ghostmpc.benchmark.runner import (
            _guard_compilation,
            _record_compilation,
        )

        with TemporaryDirectory() as directory:
            source = Path(directory) / "program.mpc"
            source.write_text("REQUESTS = 2\n")
            _record_compilation(source)
            _guard_compilation(source, skip_compile=True)  # unchanged: allowed
            source.write_text("REQUESTS = 5\n")
            with self.assertRaises(ValueError) as raised:
                _guard_compilation(source, skip_compile=True)
        self.assertIn("unchanged", str(raised.exception))

    def test_an_uncompiled_source_is_refused(self):
        from ghostmpc.benchmark.runner import _guard_compilation

        with TemporaryDirectory() as directory:
            source = Path(directory) / "program.mpc"
            source.write_text("REQUESTS = 2\n")
            with self.assertRaises(FileNotFoundError):
                _guard_compilation(source, skip_compile=True)

    def test_the_guard_is_inert_without_the_flag(self):
        from ghostmpc.benchmark.runner import _guard_compilation

        with TemporaryDirectory() as directory:
            source = Path(directory) / "program.mpc"
            source.write_text("REQUESTS = 2\n")
            _guard_compilation(source, skip_compile=False)


class ReportRecordsTheSetting(unittest.TestCase):
    def test_the_report_states_the_launch_arguments(self):
        spec = get_protocol("ps-rep-ring-party")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_report([_trial(spec.name)], _manifest(), output, "commit",
                         protocol_runtime_args=spec.runtime_args_for(32))
            report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["security"]["protocol_runtime_args"],
                         ["-R", "32", "-b", "1000000"])
        self.assertEqual(report["network"]["profile"], "co-host")


if __name__ == "__main__":
    unittest.main()
