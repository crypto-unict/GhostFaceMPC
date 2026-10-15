import tempfile
import unittest
from pathlib import Path

from ghostmpc.core.pools import PoolLedger


class PoolTests(unittest.TestCase):
    def test_pool_can_only_be_consumed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PoolLedger(Path(directory))
            pool = ledger.create("rep3", "program", 35)
            consumed = ledger.consume(pool.pool_id)
            self.assertIsNotNone(consumed.consumed_at)
            with self.assertRaises(RuntimeError):
                ledger.consume(pool.pool_id)

    def test_failed_pool_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PoolLedger(Path(directory))
            pool = ledger.create("rep3", "program", 1)
            failed = ledger.invalidate(pool.pool_id, "test failure")
            self.assertTrue(failed.exists())
            with self.assertRaises(RuntimeError):
                ledger.consume(pool.pool_id)


if __name__ == "__main__":
    unittest.main()
