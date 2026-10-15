"""Crash-safe accounting for one-time preprocessing pools."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PoolRecord:
    pool_id: str
    protocol: str
    program: str
    requests: int
    created_at: str
    consumed_at: str | None = None


class PoolLedger:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, protocol: str, program: str, requests: int) -> PoolRecord:
        record = PoolRecord(
            pool_id=uuid.uuid4().hex,
            protocol=protocol,
            program=program,
            requests=requests,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write_new(record)
        return record

    def consume(self, pool_id: str) -> PoolRecord:
        source = self.directory / f"{pool_id}.ready.json"
        consumed = self.directory / f"{pool_id}.consumed.json"
        if consumed.exists():
            raise RuntimeError(f"preprocessing pool {pool_id} has already been consumed")
        try:
            data = json.loads(source.read_text())
        except FileNotFoundError as exc:
            raise RuntimeError(f"unknown or unavailable preprocessing pool {pool_id}") from exc
        record = PoolRecord(**data)
        final = PoolRecord(
            pool_id=record.pool_id, protocol=record.protocol,
            program=record.program, requests=record.requests,
            created_at=record.created_at,
            consumed_at=datetime.now(timezone.utc).isoformat(),
        )
        fd = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(final), handle, indent=2, sort_keys=True)
            handle.write("\n")
        source.unlink()
        return final

    def invalidate(self, pool_id: str, reason: str) -> Path:
        source = self.directory / f"{pool_id}.ready.json"
        failed = self.directory / f"{pool_id}.failed.json"
        try:
            data = json.loads(source.read_text())
        except FileNotFoundError as exc:
            raise RuntimeError(f"unknown or unavailable preprocessing pool {pool_id}") from exc
        data["failed_at"] = datetime.now(timezone.utc).isoformat()
        data["reason"] = reason
        fd = os.open(failed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        source.unlink()
        return failed

    def _write_new(self, record: PoolRecord) -> None:
        path = self.directory / f"{record.pool_id}.ready.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(record), handle, indent=2, sort_keys=True)
            handle.write("\n")
