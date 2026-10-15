import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ghostmpc.benchmark import network
from ghostmpc.benchmark.runner import _protocol_commands
from ghostmpc.core.mpspdz_protocols import get_protocol

REP3 = get_protocol("replicated-ring-party")


class ProfileTests(unittest.TestCase):
    def test_shaped_profiles_carry_the_deployed_link_conditions(self):
        # A commodity gigabit segment and a provisioned wide-area link, not the
        # line rate of a network card.
        lan = network.get_profile("lan")
        wan = network.get_profile("wan")
        self.assertEqual((lan.rtt_ms, lan.bandwidth_mbit), (0.1, 1_000.0))
        self.assertEqual((wan.rtt_ms, wan.bandwidth_mbit), (100.0, 100.0))
        self.assertIsNone(network.get_profile(network.CO_HOST))

    def test_fast_wan_moves_latency_alone(self):
        # The point of the profile: the same capacity as `lan`, so what a row
        # measured over it costs above a LAN row is wide-area latency and
        # nothing else.
        lan = network.get_profile("lan")
        fast_wan = network.get_profile("fast-wan")
        self.assertEqual(
            (fast_wan.rtt_ms, fast_wan.bandwidth_mbit), (10.0, 1_000.0))
        self.assertEqual(fast_wan.bandwidth_mbit, lan.bandwidth_mbit)
        self.assertEqual(fast_wan.one_way_delay_ms, 5.0)
        # 1 Gbit/s over 10 ms is the same 1.25 MB in flight as the WAN
        # profile's 100 Mbit/s over 100 ms, hence the same queue.
        self.assertAlmostEqual(
            fast_wan.bandwidth_delay_product_bytes, 1_250_000.0)
        self.assertEqual(network.netem_limit_packets(fast_wan), 3333)

    def test_only_the_sender_egress_is_shaped(self):
        # netem sits on one egress per direction, so half the round trip each.
        self.assertEqual(network.get_profile("wan").one_way_delay_ms, 50.0)
        self.assertEqual(network.get_profile("lan").one_way_delay_ms, 0.05)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown network profile"):
            network.get_profile("satellite")

    def test_queue_covers_the_bandwidth_delay_product(self):
        wan = network.get_profile("wan")
        # 100 Mbit/s over 100 ms is 1.25 MB in flight; four of those products
        # is 3333 packets, well above netem's own default of 1000.
        self.assertAlmostEqual(wan.bandwidth_delay_product_bytes, 1_250_000.0)
        limit = network.netem_limit_packets(wan)
        self.assertEqual(limit, 3333)
        self.assertGreater(limit * 1500, wan.bandwidth_delay_product_bytes)

    def test_queue_never_falls_below_the_netem_default(self):
        # 1 Gbit/s over 0.1 ms is only 12.5 kB, so four products is 33 packets;
        # the floor is what keeps the LAN queue able to absorb a bulk burst.
        lan = network.get_profile("lan")
        self.assertAlmostEqual(lan.bandwidth_delay_product_bytes, 12_500.0)
        self.assertEqual(network.netem_limit_packets(lan), 1000)
        thin = network.NetworkProfile("thin", rtt_ms=0.1, bandwidth_mbit=1.0)
        self.assertEqual(network.netem_limit_packets(thin), 1000)


class AddressTests(unittest.TestCase):
    def test_addresses_and_namespaces_are_stable(self):
        self.assertEqual(
            [network.party_address(i) for i in range(3)],
            ["10.77.0.10", "10.77.0.11", "10.77.0.12"])
        self.assertEqual(network.namespace(1), "ghost-p1")
        self.assertEqual(network.interface(2), "p2")

    def test_hosts_file_is_one_address_per_line(self):
        # MP-SPDZ Names::init reads exactly this and derives port base + i.
        self.assertEqual(
            network.hosts_file_contents().splitlines(),
            ["10.77.0.10", "10.77.0.11", "10.77.0.12"])


class TopologyTests(unittest.TestCase):
    def test_wan_topology_is_the_expected_iproute2_sequence(self):
        commands = [" ".join(c) for c in network.topology_commands(
            network.get_profile("wan"))]
        self.assertEqual(commands[:2], [
            "ip link add br0 type bridge", "ip link set br0 up"])
        self.assertEqual(commands[2:11], [
            "ip netns add ghost-p0",
            "ip link add p0 type veth peer name b0",
            "ip link set b0 master br0",
            "ip link set b0 up",
            "ip link set p0 netns ghost-p0",
            "ip -n ghost-p0 addr add 10.77.0.10/24 dev p0",
            "ip -n ghost-p0 link set p0 up",
            "ip -n ghost-p0 link set lo up",
            "ip netns exec ghost-p0 tc qdisc add dev p0 root netem "
            "delay 50.0ms rate 100mbit limit 3333",
        ])
        self.assertEqual(len(commands), 2 + 3 * 9)

    def test_co_host_builds_the_same_paths_without_a_qdisc(self):
        # The control keeps the veth path so only the shaping differs.
        commands = network.topology_commands(None)
        self.assertEqual(len(commands), 2 + 3 * 8)
        self.assertFalse(any("tc" in c for c in commands))

    def test_jitter_and_loss_reach_the_qdisc(self):
        profile = network.NetworkProfile(
            "lossy", rtt_ms=20.0, bandwidth_mbit=100.0,
            jitter_ms=2.0, loss_percent=0.5)
        qdisc = [c for c in network.topology_commands(profile, parties=1)
                 if "netem" in c][0]
        self.assertEqual(
            qdisc[qdisc.index("netem"):],
            ["netem", "delay", "10.0ms", "2.0ms", "loss", "0.5%",
             "rate", "100mbit", "limit", "1000"])


class ContextTests(unittest.TestCase):
    def test_context_is_absent_outside_the_wrapper(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(network.active_context())
            self.assertIsNone(network.require_context(network.CO_HOST))

    def test_shaped_profile_outside_the_wrapper_names_the_remedy(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ghostmpc-netsim wan"):
                network.require_context("wan")

    def test_half_prepared_environment_is_an_error(self):
        with mock.patch.dict(os.environ, {network.PROFILE_ENVIRONMENT: "wan"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "did not finish preparing"):
                network.active_context()

    def test_requesting_a_profile_the_network_does_not_have_is_an_error(self):
        prepared = {network.PROFILE_ENVIRONMENT: "lan",
                    network.HOSTS_ENVIRONMENT: "/run/hosts"}
        with mock.patch.dict(os.environ, prepared, clear=True):
            with self.assertRaisesRegex(RuntimeError, "profile mismatch"):
                network.require_context("wan")
            self.assertEqual(network.require_context("lan").profile_name, "lan")


class CommandTests(unittest.TestCase):
    def test_the_co_host_argv_is_unchanged(self):
        commands = _protocol_commands(
            Path("/mpspdz"), REP3, "prog", 5000, 32)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0], [
            "/mpspdz/replicated-ring-party.x", "0", "prog",
            "-pn", "5000", "-h", "localhost", "-R", "32", "-v"])

    def test_shaped_argv_enters_the_namespace_and_uses_the_hosts_file(self):
        context = network.NetworkContext(
            network.get_profile("wan"), Path("/run/hosts"))
        commands = _protocol_commands(
            Path("/mpspdz"), REP3, "prog", 5000, 32, context)
        for party, command in enumerate(commands):
            self.assertEqual(
                command[:4], ["ip", "netns", "exec", f"ghost-p{party}"])
            # -h would send every party to one loopback address.
            self.assertNotIn("-h", command)
            self.assertEqual(command[command.index("-ip") + 1], "/run/hosts")


class ParserTests(unittest.TestCase):
    PING = (
        "PING 10.77.0.11 (10.77.0.11) 56(84) bytes of data.\n\n"
        "--- 10.77.0.11 ping statistics ---\n"
        "3 packets transmitted, 3 received, 0% packet loss, time 2002ms\n"
        "rtt min/avg/max/mdev = 20.095/26.794/40.174/9.460 ms\n")
    QDISC = json.dumps([{
        "kind": "netem", "handle": "8003:", "root": True,
        "options": {"limit": 3333},
        "bytes": 70169550, "packets": 46371, "drops": 0,
        "overlimits": 0, "requeues": 0,
    }])

    def _context(self):
        return network.NetworkContext(
            network.get_profile("wan"), Path("/run/hosts"), parties=2)

    def test_rtt_is_read_for_every_ordered_pair(self):
        with mock.patch.object(network, "_run", return_value=self.PING) as run:
            measured = network.measure_rtt(self._context(), count=3)
        self.assertEqual(sorted(measured), ["0->1", "1->0"])
        # The minimum is the link; the mean also pays neighbour discovery.
        self.assertEqual(measured["0->1"]["min_ms"], 20.095)
        self.assertEqual(measured["0->1"]["avg_ms"], 26.794)
        self.assertEqual(run.call_args_list[0].args[0][:4],
                         ["ip", "netns", "exec", "ghost-p0"])

    def test_unreadable_ping_output_is_an_error(self):
        with mock.patch.object(network, "_run", return_value="Network is unreachable"):
            with self.assertRaisesRegex(RuntimeError, "ping statistics"):
                network.measure_rtt(self._context(), count=1)

    def test_link_counters_read_the_root_qdisc(self):
        with mock.patch.object(network, "_run", return_value=self.QDISC):
            counters = network.link_counters(self._context())
        self.assertEqual(len(counters), 2)
        self.assertEqual(counters[0],
                         {"bytes": 70169550, "packets": 46371, "drops": 0,
                          "requeues": 0, "overlimits": 0})

    def test_missing_root_qdisc_is_an_error(self):
        with mock.patch.object(network, "_run", return_value="[]"):
            with self.assertRaisesRegex(RuntimeError, "no root qdisc"):
                network.link_counters(self._context())

    def test_the_unshaped_control_reports_no_link_statistics(self):
        # It carries no netem qdisc on purpose, so that it stays a faithful
        # reproduction of the deployed baseline.
        control = network.NetworkContext(None, Path("/run/hosts"))
        with mock.patch.object(network, "_run") as run:
            self.assertIsNone(network.link_counters(control))
        run.assert_not_called()


class ReportTests(unittest.TestCase):
    def _block(self, **kwargs):
        context = network.NetworkContext(
            network.get_profile("wan"), Path("/run/hosts"))
        with mock.patch.object(network, "iproute2_version", return_value="iproute2-6.15.0"):
            return network.network_report(context, **kwargs)

    def test_a_run_outside_the_wrapper_declares_no_emulation(self):
        block = network.network_report(None)
        self.assertEqual(block["profile"], network.CO_HOST)
        self.assertEqual(block["emulation"], "none")
        self.assertFalse(block["physical_hosts"])

    def test_the_control_is_neither_shaped_nor_an_ordinary_loopback_run(self):
        control = network.NetworkContext(None, Path("/run/hosts"))
        block = network.network_report(control)
        self.assertEqual(block["emulation"], "namespaces-unshaped")
        self.assertIsNone(block["tool"])
        self.assertIsNone(block["configured"])

    def test_a_shaped_run_is_labelled_simulated_and_records_its_topology(self):
        # The plans require every network figure to say whether it is physical
        # or simulated; this path can only produce the latter.
        block = self._block()
        self.assertEqual(block["emulation"], "simulated")
        self.assertFalse(block["physical_hosts"])
        self.assertIn("netem", block["tool"])
        self.assertEqual(block["configured"]["rtt_ms"], 100.0)
        self.assertEqual(len(block["topology_commands"]), 2 + 3 * 9)

    def test_framing_ratio_carries_over_to_the_per_inference_figure(self):
        block = self._block(
            wire_bytes_by_party=[1_046_000, 1_046_000, 1_046_000],
            drops_by_party=[0, 0, 0],
            payload_bytes=3_000_000,
            per_inference_payload_bytes=476_275_886,
            per_inference_wall_seconds=45.0)
        self.assertEqual(block["wire_bytes_total"], 3_138_000)
        self.assertAlmostEqual(block["wire_overhead_ratio"], 1.046)
        self.assertAlmostEqual(
            block["wire_bytes_per_inference"], 476_275_886 * 1.046)
        self.assertAlmostEqual(
            block["achieved_mbit_per_second"],
            476_275_886 * 1.046 * 8.0 / 45.0 / 1e6)
        # Three parties each capped at the profile bandwidth.
        self.assertAlmostEqual(
            block["link_utilisation"],
            block["achieved_mbit_per_second"] / 300.0)

    def test_partial_observations_do_not_invent_a_ratio(self):
        block = self._block(wire_bytes_by_party=[1, 2, 3])
        self.assertEqual(block["wire_bytes_total"], 6)
        self.assertNotIn("wire_overhead_ratio", block)


if __name__ == "__main__":
    unittest.main()
