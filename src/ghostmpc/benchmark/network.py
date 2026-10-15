"""Emulated LAN/WAN links for the three-party benchmark harness.

The three parties normally run as processes on one host and talk over the
loopback interface, where a round costs no propagation delay and the byte
count is the only thing a report can honestly claim. This module places each
party in its own network namespace behind a `netem` qdisc, so a measured wall
time carries a declared round-trip time and a declared link capacity.

Nothing here emulates a network. The emulation is `tc netem` from iproute2;
this module only composes its arguments, in one place, so the emitter, the
wrapper script and the report cannot disagree about the topology that ran.

Layout, for `parties = 3`:

    ghost-p0            ghost-p1            ghost-p2         network namespaces
      p0                  p1                  p2             party interfaces
       \\                  |                  /
        b0 --------------- br0 ------------- b2              bridge, outer ns

`netem` sits on each party interface's egress and nowhere else. A packet from
party i to party j is therefore delayed once, by `rtt/2`, and the reply is
delayed once more, so the observed pairwise round-trip time is exactly `rtt`.
The `rate` on the same qdisc bounds what one party can push, which is the
per-server network card of a real deployment; the bridge is not shaped, so it
contributes no second bottleneck.

Shaping the loopback interface instead was rejected: a single qdisc on `lo`
applies to all three parties at once and delays every hop of a message twice.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# The parties live on their own subnet inside the namespaces, so no address
# here can collide with anything the host routes.
SUBNET = ipaddress.ip_network("10.77.0.0/24")
_FIRST_HOST = 10
BRIDGE = "br0"
NAMESPACE_PREFIX = "ghost-p"

PROFILE_ENVIRONMENT = "GHOSTMPC_NETSIM_PROFILE"
HOSTS_ENVIRONMENT = "GHOSTMPC_NETSIM_HOSTS"

# An Ethernet frame carrying a full segment. Used only to turn the
# bandwidth-delay product into the packet unit netem's `limit` expects.
_MTU_BYTES = 1500
# netem's own default is 1000 packets, which is below the bandwidth-delay
# product of the WAN profile and would drop the protocol's bulk transfers.
_MINIMUM_LIMIT_PACKETS = 1000
# Queue four bandwidth-delay products, so a burst that momentarily exceeds the
# in-flight window is delayed rather than discarded.
_LIMIT_BDP_MULTIPLE = 4


@dataclass(frozen=True)
class NetworkProfile:
    """One declared link condition, applied identically to every party."""

    name: str
    rtt_ms: float
    bandwidth_mbit: float
    jitter_ms: float = 0.0
    loss_percent: float = 0.0

    @property
    def one_way_delay_ms(self) -> float:
        """The delay netem applies on a single egress.

        Only the sender's egress is shaped, so one traversal costs half the
        round trip and the pairwise round-trip time comes out as `rtt_ms`.
        """
        return self.rtt_ms / 2.0

    @property
    def bandwidth_delay_product_bytes(self) -> float:
        return self.bandwidth_mbit * 1e6 * (self.rtt_ms / 1000.0) / 8.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "rtt_ms": self.rtt_ms,
            "bandwidth_mbit": self.bandwidth_mbit,
            "jitter_ms": self.jitter_ms, "loss_percent": self.loss_percent,
            "one_way_delay_ms": self.one_way_delay_ms,
            "netem_limit_packets": netem_limit_packets(self),
        }


# `co-host` is the unshaped control: it keeps the namespaces and the bridge so
# that the veth path itself is measured, and applies no qdisc.
CO_HOST = "co-host"
NETWORK_PROFILES: dict[str, NetworkProfile | None] = {
    CO_HOST: None,
    # A commodity gigabit datacentre segment, and a wide-area link at the rate
    # a real inter-datacentre or inter-institution deployment is provisioned
    # at rather than the line rate of its network card.
    "lan": NetworkProfile("lan", rtt_ms=0.1, bandwidth_mbit=1_000.0),
    "wan": NetworkProfile("wan", rtt_ms=100.0, bandwidth_mbit=100.0),
    # `wan` moves latency and capacity at once, so a row measured over it
    # prices the two together and neither separately. `fast-wan` holds the
    # capacity at the LAN's rate and moves only the round-trip time, by two
    # orders of magnitude: read against `lan` it is the price of wide-area
    # latency alone, read against `wan` it is what the narrower link adds on
    # top. It is also the regime a regional deployment actually runs in,
    # between a co-located segment and an intercontinental one.
    "fast-wan": NetworkProfile("fast-wan", rtt_ms=10.0, bandwidth_mbit=1_000.0),
}


def get_profile(name: str) -> NetworkProfile | None:
    if name not in NETWORK_PROFILES:
        raise ValueError(
            f"unknown network profile {name!r}; "
            f"choose one of {', '.join(sorted(NETWORK_PROFILES))}")
    return NETWORK_PROFILES[name]


def netem_limit_packets(profile: NetworkProfile) -> int:
    """Queue length in packets, sized from the bandwidth-delay product."""
    packets = profile.bandwidth_delay_product_bytes / _MTU_BYTES
    return max(_MINIMUM_LIMIT_PACKETS, int(_LIMIT_BDP_MULTIPLE * packets))


def party_address(party: int) -> str:
    return str(SUBNET.network_address + _FIRST_HOST + party)


def namespace(party: int) -> str:
    return f"{NAMESPACE_PREFIX}{party}"


def interface(party: int) -> str:
    return f"p{party}"


def _netem_arguments(profile: NetworkProfile) -> list[str]:
    arguments = ["delay", f"{profile.one_way_delay_ms}ms"]
    if profile.jitter_ms:
        arguments.append(f"{profile.jitter_ms}ms")
    if profile.loss_percent:
        arguments += ["loss", f"{profile.loss_percent}%"]
    return arguments + [
        "rate", f"{profile.bandwidth_mbit:g}mbit",
        "limit", str(netem_limit_packets(profile)),
    ]


def topology_commands(profile: NetworkProfile | None,
                      parties: int = 3) -> list[list[str]]:
    """The exact iproute2 argv sequence that builds the emulated network.

    Returned rather than executed so the wrapper script can run it and the
    report can record it verbatim: what a reader checks is this list, not a
    description of it.
    """
    commands: list[list[str]] = [
        ["ip", "link", "add", BRIDGE, "type", "bridge"],
        ["ip", "link", "set", BRIDGE, "up"],
    ]
    for party in range(parties):
        name, device, peer = namespace(party), interface(party), f"b{party}"
        prefix = ["ip", "netns", "exec", name]
        commands += [
            ["ip", "netns", "add", name],
            ["ip", "link", "add", device, "type", "veth", "peer", "name", peer],
            ["ip", "link", "set", peer, "master", BRIDGE],
            ["ip", "link", "set", peer, "up"],
            ["ip", "link", "set", device, "netns", name],
            ["ip", "-n", name, "addr", "add",
             f"{party_address(party)}/{SUBNET.prefixlen}", "dev", device],
            ["ip", "-n", name, "link", "set", device, "up"],
            ["ip", "-n", name, "link", "set", "lo", "up"],
        ]
        if profile is not None:
            commands.append(
                prefix + ["tc", "qdisc", "add", "dev", device, "root", "netem",
                          *_netem_arguments(profile)])
    return commands


def hosts_file_contents(parties: int = 3) -> str:
    """The `-ip` file MP-SPDZ reads: one party address per line.

    `Names::init` assigns party i the port `portnum_base + i`, so the port is
    left implicit and the harness keeps passing `-pn`.
    """
    return "".join(f"{party_address(party)}\n" for party in range(parties))


class NetworkContext:
    """A prepared emulated network, as seen from inside the wrapper."""

    def __init__(self, profile: NetworkProfile | None, hosts_file: Path,
                 parties: int = 3) -> None:
        self.profile = profile
        self.hosts_file = hosts_file
        self.parties = parties

    @property
    def profile_name(self) -> str:
        return self.profile.name if self.profile is not None else CO_HOST

    def party_prefix(self, party: int) -> list[str]:
        return ["ip", "netns", "exec", namespace(party)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NetworkContext(profile={self.profile_name!r}, parties={self.parties})"


def active_context(parties: int = 3) -> NetworkContext | None:
    """The context the wrapper script prepared, or None outside it."""
    name = os.environ.get(PROFILE_ENVIRONMENT)
    if not name:
        return None
    hosts = os.environ.get(HOSTS_ENVIRONMENT)
    if not hosts:
        raise RuntimeError(
            f"{PROFILE_ENVIRONMENT} is set but {HOSTS_ENVIRONMENT} is not; "
            "the netsim wrapper did not finish preparing the network")
    return NetworkContext(get_profile(name), Path(hosts), parties=parties)


def require_context(requested: str, parties: int = 3) -> NetworkContext | None:
    """Resolve the requested profile against the prepared network.

    Building the namespaces needs CAP_NET_ADMIN, which the harness does not
    hold and must not acquire on its own. A shaped run therefore only makes
    sense inside `scripts/netsim/ghostmpc-netsim`, and asking for one outside
    it is an error with a copyable remedy rather than a silent co-host run.
    """
    get_profile(requested)
    context = active_context(parties=parties)
    if context is None:
        if requested == CO_HOST:
            return None
        raise RuntimeError(
            f"network profile {requested!r} needs the emulated network; run\n"
            f"    scripts/netsim/ghostmpc-netsim {requested} -- <command>")
    if context.profile_name != requested:
        raise RuntimeError(
            f"network profile mismatch: the prepared network is "
            f"{context.profile_name!r} but {requested!r} was requested")
    return context


# --- measurement of the emulated network itself -----------------------------

def _tc_binary() -> str:
    # tc ships in /sbin, which is not on a non-root PATH on Debian.
    return shutil.which("tc") or "/sbin/tc"


def _run(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), check=True, capture_output=True, text=True).stdout


_PING_STATISTICS = re.compile(
    r"^rtt min/avg/max/mdev = "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+) ms", re.MULTILINE)


def _ordered_pairs(parties: int) -> Iterator[tuple[int, int]]:
    for source in range(parties):
        for target in range(parties):
            if source != target:
                yield source, target


# ping's default one second interval would make six ordered pairs take two
# minutes. 0.2 s is the shortest an unprivileged process may ask for.
_PING_INTERVAL_SECONDS = 0.2


def measure_rtt(context: NetworkContext, count: int = 10
                ) -> dict[str, dict[str, float]]:
    """Ping every ordered pair and report what the link actually delivers.

    The minimum is the figure to compare against the profile: the first packet
    of a pair also pays neighbour discovery, which inflates the mean.
    """
    measured: dict[str, dict[str, float]] = {}
    for source, target in _ordered_pairs(context.parties):
        output = _run(context.party_prefix(source) + [
            "ping", "-c", str(count), "-i", str(_PING_INTERVAL_SECONDS),
            "-q", party_address(target)])
        match = _PING_STATISTICS.search(output)
        if match is None:
            raise RuntimeError(
                f"could not read ping statistics for {source}->{target}:\n{output}")
        minimum, average, maximum, deviation = (float(x) for x in match.groups())
        measured[f"{source}->{target}"] = {
            "min_ms": minimum, "avg_ms": average,
            "max_ms": maximum, "mdev_ms": deviation}
    return measured


def link_counters(context: NetworkContext) -> list[dict[str, int]] | None:
    """Wire bytes, packets and drops on each party's shaped egress.

    These are the bytes that crossed the emulated link, so unlike the counters
    MP-SPDZ reports they include TCP, TLS, IP and Ethernet framing.

    None under the unshaped control: it carries no netem qdisc, and the default
    `noqueue` on a veth keeps no statistics. Leaving the control unshaped keeps
    it the faithful reproduction of the deployed baseline that it is there to
    be, which is worth more than counters nothing else needs.
    """
    if context.profile is None:
        return None
    counters: list[dict[str, int]] = []
    for party in range(context.parties):
        output = _run(context.party_prefix(party) + [
            _tc_binary(), "-s", "-j", "qdisc", "show", "dev", interface(party)])
        root = next(
            (entry for entry in json.loads(output) if entry.get("root")), None)
        if root is None:
            raise RuntimeError(f"no root qdisc on {interface(party)}")
        counters.append({
            "bytes": int(root.get("bytes", 0)),
            "packets": int(root.get("packets", 0)),
            "drops": int(root.get("drops", 0)),
            "requeues": int(root.get("requeues", 0)),
            "overlimits": int(root.get("overlimits", 0)),
        })
    return counters


def iproute2_version() -> str:
    try:
        return _run(["ip", "-V"]).strip()
    except Exception:  # pragma: no cover - only in stripped environments
        return "unknown"


def network_report(context: NetworkContext | None, *,
                   measured_rtt: dict[str, dict[str, float]] | None = None,
                   wire_bytes_by_party: Sequence[int] | None = None,
                   drops_by_party: Sequence[int] | None = None,
                   payload_bytes: int | None = None,
                   per_inference_payload_bytes: int | None = None,
                   per_inference_wall_seconds: float | None = None
                   ) -> dict[str, object]:
    """The `network` block of report.json.

    `emulation` is always `simulated` and `physical_hosts` always false: the
    plans require every network figure to say which of the two it is, and this
    path can only ever produce the former.

    The qdisc counters cover whole process groups, warm-ups and handshakes
    included, so they are not divided by a request count. They are turned into
    a framing ratio against the payload MP-SPDZ reports for that very same
    span, and the ratio is what carries over to the per-inference figure.
    """
    profile = context.profile if context is not None else None
    if context is None:
        emulation = "none"
    elif profile is None:
        # The control ran in the namespaces but through no qdisc, so it is
        # neither a shaped measurement nor an ordinary loopback run.
        emulation = "namespaces-unshaped"
    else:
        emulation = "simulated"
    block: dict[str, object] = {
        "profile": context.profile_name if context is not None else CO_HOST,
        "emulation": emulation,
        "tool": f"tc netem ({iproute2_version()})" if profile is not None else None,
        "physical_hosts": False,
        "topology": "three network namespaces bridged by br0; netem on each party egress",
        "configured": profile.as_dict() if profile is not None else None,
        "topology_commands": (
            [" ".join(command) for command in topology_commands(
                profile, context.parties)] if context is not None else []),
        "measured_rtt_ms": measured_rtt,
        "wire_bytes_by_party": list(wire_bytes_by_party) if wire_bytes_by_party else None,
        "drops_by_party": list(drops_by_party) if drops_by_party else None,
        "definition": {
            "wire_bytes": ("bytes leaving each party's shaped egress over every "
                           "measured process group, from the netem qdisc counters; "
                           "includes TCP, TLS, IP and Ethernet framing"),
            "wire_overhead_ratio": "wire bytes divided by the MP-SPDZ payload of the same span",
            "measured_rtt_ms": "ping between every ordered pair; compare the minimum, the mean also pays neighbour discovery",
        },
    }
    if not wire_bytes_by_party:
        return block
    total = int(sum(wire_bytes_by_party))
    block["wire_bytes_total"] = total
    if not payload_bytes:
        return block
    ratio = total / float(payload_bytes)
    block["payload_bytes_total"] = int(payload_bytes)
    block["wire_overhead_ratio"] = ratio
    if per_inference_payload_bytes:
        per_inference_wire = per_inference_payload_bytes * ratio
        block["payload_bytes_per_inference"] = int(per_inference_payload_bytes)
        block["wire_bytes_per_inference"] = per_inference_wire
        if per_inference_wall_seconds:
            achieved = per_inference_wire * 8.0 / per_inference_wall_seconds / 1e6
            block["achieved_mbit_per_second"] = achieved
            if profile is not None:
                # The three parties send in parallel, each capped at the
                # profile bandwidth, so the aggregate ceiling is three links.
                block["link_utilisation"] = achieved / (
                    profile.bandwidth_mbit * context.parties)
    return block


def emit(argv: Sequence[str]) -> int:
    """The wrapper script's view of this module: `topology <profile>`/`hosts`.

    The shell holds no knowledge of the network beyond running the lines it is
    handed, so the topology has exactly one definition.
    """
    if argv[:1] == ["hosts"]:
        print(hosts_file_contents(), end="")
        return 0
    if len(argv) == 2 and argv[0] == "topology":
        for command in topology_commands(get_profile(argv[1])):
            print(" ".join(command))
        return 0
    print("usage: emit topology <profile> | emit hosts")
    return 2
