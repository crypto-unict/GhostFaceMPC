"""The two MP-SPDZ protocols the program is emitted for.

Both are honest-majority three-party protocols over the replicated secret
sharing of Araki et al. on the ring ``Z_{2^32}``: ``replicated-ring-party`` is
secure against one semi-honest server, ``ps-rep-ring-party`` against one
actively corrupted server, with abort (post-sacrifice multiplication check).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    name: str
    binary: str
    compile_args: tuple[str, ...]
    runtime_args: tuple[str, ...]
    parties: int
    input_parties: int
    security: str = "semi-honest"
    #: The repository's ``prefetchprep`` VM instruction over an actively secure
    #: live preprocessing: it fills, before a request, every buffer the request
    #: draws on -- the random bits a daBit is made from and the binary triples
    #: of the AND gates.  The semi-honest protocol uses ``prefetchdabit``, which
    #: fills the daBit buffer alone.
    prep_prefetch: bool = False
    #: The two-summand regrouping of the rounding kernel (``shrsi2`` and
    #: ``split_to_n_summands(k, 2)``).  It hands ``A = x_0 + x_2`` to P0 alone,
    #: so the re-shared summand has no second holder and a corrupt P0 could
    #: inject an undetected additive error: only a semi-honest protocol may take
    #: it, and ``PostSacriRepRingShare::split`` refuses it upstream as well.
    two_summand_split: bool = False
    #: The protocol object accumulates the cross terms of an inner product
    #: *before* the exchange, re-sharing one ring element per output instead of
    #: one per term (``Protocols/Replicated.h``).  The post-sacrifice protocol
    #: inherits ``ProtocolBase``'s unfused default.
    fused_dot_product: bool = False
    #: Statistical security parameter of an actively secure protocol, mirroring
    #: MP-SPDZ's ``DEFAULT_SECURITY``.
    security_parameter: int | None = None
    #: Runtime arguments this profile is deployed with, appended after the
    #: spec's own.  They tune the protocol; they never change what it computes.
    runtime_defaults: tuple[str, ...] = ()

    @property
    def malicious(self) -> bool:
        return self.security != "semi-honest"

    @property
    def prefetch_instruction(self) -> str:
        """The VM instruction that fills one request's preprocessing."""
        return "prefetchprep" if self.prep_prefetch else "prefetchdabit"

    def compile_args_for(self, ring_bits: int) -> tuple[str, ...]:
        """Compilation arguments carrying the ring width the profile asks for.

        The width is *substituted* rather than appended, so a profile's ring
        width has exactly one source of truth: the manifest.
        """
        return self._with_ring(self.compile_args, ring_bits)

    def runtime_args_for(self, ring_bits: int) -> tuple[str, ...]:
        """Party arguments carrying the same ring width as the compilation."""
        return self._with_ring(self.runtime_args, ring_bits) + self.runtime_defaults

    @staticmethod
    def _with_ring(args: tuple[str, ...], ring_bits: int) -> tuple[str, ...]:
        resolved = list(args)
        for index, value in enumerate(resolved[:-1]):
            if value == "-R":
                resolved[index + 1] = str(ring_bits)
        return tuple(resolved)

    @property
    def source_directives(self) -> str:
        # Every truncation is exact (no probabilistic truncation), and the
        # comparison and rounding kernels read the three-way local split of the
        # replicated sharing.
        return "\n".join(["program.use_trunc_pr = False", "program.use_split(3)"])

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROTOCOLS: dict[str, ProtocolSpec] = {spec.name: spec for spec in (
    ProtocolSpec(
        "replicated-ring-party", "replicated-ring-party.x",
        ("-E", "ring", "-R", "32"), ("-R", "32"), 3, 3,
        two_summand_split=True, fused_dot_product=True),
    # PostSacriRepRingShare<K,S> is Rep3Share2<K> with malicious = true, a
    # post-sacrifice multiplication check and HashMaliciousRepMC, so the share
    # layout -- and with it the three-way split and the local summand shift the
    # bounded truncation is built on -- is the semi-honest one.
    ProtocolSpec(
        "ps-rep-ring-party", "ps-rep-ring-party.x",
        ("-E", "ps-rep-ring", "-R", "32"), ("-R", "32"), 3, 3,
        security="malicious-abort", security_parameter=40, prep_prefetch=True,
        # A preprocessing batch of 1,000,000 instead of MP-SPDZ's default
        # 10,000 cuts the rounds of live preprocessing under active security by
        # more than an order of magnitude and changes no output bit.
        runtime_defaults=("-b", "1000000")),
)}


def get_protocol(name: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        choices = ", ".join(PROTOCOLS)
        raise ValueError(
            f"unsupported MP-SPDZ protocol {name!r}; choose one of: {choices}"
        ) from exc
