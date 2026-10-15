import unittest
from unittest import mock

from ghostmpc.cli import build_parser


class CommandLineTests(unittest.TestCase):
    def test_the_subcommands(self):
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices
        self.assertEqual(set(commands), {
            "fetch-checkpoint", "fetch-bundles", "bootstrap-mpspdz",
            "generate", "benchmark", "share-image", "share-gallery", "open-result",
            "evaluate-accuracy", "export", "export-porthos",
            "calibrate-bounds", "prepare-ms1mv3", "qat-ring"})

    def test_export_writes_the_deployment_profile(self):
        args = build_parser().parse_args(["export", "model.h5", "out"])
        manifest = mock.Mock()
        manifest.to_dict.return_value = {}
        with mock.patch("ghostmpc.cli.export.export_checkpoint",
                        return_value=manifest) as export:
            self.assertEqual(args.func(args), 0)
        quantization = export.call_args.kwargs["quantization"]
        self.assertEqual(quantization.profile, "ring32-f6-fw9")
        self.assertEqual(quantization.ring_bits, 32)
        self.assertEqual(quantization.activation_bits, 15)
        self.assertEqual(quantization.fractional_bits, 6)
        self.assertEqual(quantization.weight_fractional_bits, 9)

    def test_benchmark_defaults_are_the_paper_design(self):
        args = build_parser().parse_args(["benchmark"])
        self.assertEqual(args.manifest, "artifacts/model/manifest.json")
        self.assertEqual(args.protocol, "replicated-ring-party")
        self.assertEqual((args.pools, args.warmups, args.trials), (3, 5, 30))
        self.assertEqual(args.network_profile, "co-host")
        self.assertFalse(args.score or args.match or args.identify)

    def test_verification_heads(self):
        parser = build_parser()
        score = parser.parse_args(["benchmark", "--score", "--gallery-rows", "1024"])
        self.assertTrue(score.score)
        self.assertEqual(score.gallery_fractional_bits, 15)
        identify = parser.parse_args(["benchmark", "--identify", "--gallery-rows", "128"])
        self.assertTrue(identify.identify)
        self.assertIsNone(identify.identify_candidates)
        match = parser.parse_args(["generate", "m.json", "o.mpc", "--score",
                                   "--match", "--threshold", "0.28"])
        self.assertEqual(match.threshold, 0.28)

    def test_only_the_deployed_protocols_and_networks_are_offered(self):
        parser = build_parser()
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["benchmark", "--protocol", "shamir-party"])
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["benchmark", "--network-profile", "lan-10g"])
        for network in ("co-host", "lan", "fast-wan", "wan"):
            self.assertEqual(parser.parse_args(
                ["benchmark", "--network-profile", network]).network_profile, network)

    def test_ablation_flags_do_not_exist(self):
        parser = build_parser()
        for flag in ("--bit-injection", "--probabilistic-truncation",
                     "--no-two-summand-rounding", "--packed-conv1x1",
                     "--prelu-dual-kernel", "--mpspdz-arg"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit), \
                    mock.patch("sys.stderr"):
                parser.parse_args(["benchmark", flag])

    def test_share_image_uses_deployment_scale(self):
        args = build_parser().parse_args(["share-image", "face.png", "shares"])
        self.assertEqual(args.fractional_bits, 6)
        self.assertEqual(args.ring_bits, 32)

    def test_the_client_writes_each_server_its_replicated_pair(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path

        from PIL import Image

        from ghostmpc.core.sharing import (
            encode_fixed,
            persistence_header,
            read_persistence,
            reconstruct_replicated,
        )

        parser = build_parser()

        def run(*argv):
            args = parser.parse_args(list(argv))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = args.func(args)
            return status, output.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (112, 112), (255, 0, 127)).save(root / "face.png")
            shares = root / "Persistence"
            run("share-gallery", str(shares), "--rows", "2", "--dimension", "4")
            run("share-image", str(root / "face.png"), str(shares),
                "--append", "--identity", "1")
            paths = [shares / f"Transactions-P{party}.data" for party in range(3)]
            pairs = [read_persistence(path) for path in paths]
            values = reconstruct_replicated(pairs)
            pixels = encode_fixed([1.0, -1.0, 127 / 127.5 - 1.0] * 112 * 112, 6)
            self.assertEqual(values[8:], pixels)
            for party in range(3):
                self.assertEqual(
                    (shares / f"Transactions-clear-P{party}.data").read_bytes(),
                    persistence_header(clear=True) + (1).to_bytes(4, "little"))
            # The servers append their pair of the answer to the same files.
            status, output = run("open-result", *map(str, paths), "--last", "2")
            self.assertEqual(status, 0)
            self.assertIn('"status": "accepted"', output)

    def test_qat_defaults_are_the_published_run(self):
        args = build_parser().parse_args([
            "qat-ring", "--manifest", "m.json", "--teacher", "t.h5",
            "--shards", "shards", "--output", "run"])
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.warmup_steps, 500)
        self.assertEqual(args.epochs * args.steps_per_epoch, 72_784)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.ema_decay, 0.999)


if __name__ == "__main__":
    unittest.main()
