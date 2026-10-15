import unittest
from pathlib import Path

from ghostmpc.benchmark.runner import _protocol_commands
from ghostmpc.core.mpspdz_protocols import PROTOCOLS, get_protocol


class ProtocolRegistryTests(unittest.TestCase):
    def test_the_two_deployed_protocols(self):
        self.assertEqual(set(PROTOCOLS), {"replicated-ring-party", "ps-rep-ring-party"})
        semi_honest = get_protocol("replicated-ring-party")
        malicious = get_protocol("ps-rep-ring-party")
        self.assertFalse(semi_honest.malicious)
        self.assertIsNone(semi_honest.security_parameter)
        self.assertTrue(malicious.malicious)
        self.assertEqual(malicious.security_parameter, 40)

    def test_only_the_semi_honest_ring_takes_the_two_summand_split(self):
        # The regrouping hands A = x_0 + x_2 to P0 alone, which an active
        # adversary could exploit; the malicious profile keeps three summands.
        self.assertTrue(get_protocol("replicated-ring-party").two_summand_split)
        self.assertFalse(get_protocol("ps-rep-ring-party").two_summand_split)

    def test_each_protocol_prefetches_its_own_preprocessing(self):
        self.assertEqual(get_protocol("replicated-ring-party").prefetch_instruction,
                         "prefetchdabit")
        self.assertEqual(get_protocol("ps-rep-ring-party").prefetch_instruction,
                         "prefetchprep")

    def test_the_ring_width_reaches_the_compiler_and_the_parties(self):
        spec = get_protocol("replicated-ring-party")
        self.assertEqual(spec.compile_args_for(32), ("-E", "ring", "-R", "32"))
        self.assertEqual(spec.runtime_args_for(32), ("-R", "32"))
        self.assertEqual(spec.compile_args_for(64), ("-E", "ring", "-R", "64"))
        commands = _protocol_commands(
            Path("/tmp/mp-spdz-registry-test"), spec, "program", 12345, 32)
        self.assertEqual(len(commands), 3)
        for party, command in enumerate(commands):
            self.assertEqual(command[1], str(party))
            self.assertEqual(command[command.index("-R") + 1], "32")

    def test_the_malicious_profile_is_deployed_with_a_large_batch(self):
        spec = get_protocol("ps-rep-ring-party")
        self.assertEqual(spec.compile_args_for(32), ("-E", "ps-rep-ring", "-R", "32"))
        self.assertEqual(spec.runtime_args_for(32), ("-R", "32", "-b", "1000000"))

    def test_unknown_protocols_are_refused(self):
        with self.assertRaises(ValueError):
            get_protocol("shamir-party")

    def test_both_programs_read_the_three_way_split(self):
        for spec in PROTOCOLS.values():
            self.assertIn("program.use_split(3)", spec.source_directives)
            self.assertIn("program.use_trunc_pr = False", spec.source_directives)


if __name__ == "__main__":
    unittest.main()
