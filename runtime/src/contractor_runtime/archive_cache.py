"""Optional process-owned encoded archive cache. Server metadata remains authority.

Bookkeeping is confined to one event loop: changes between awaits are atomic,
with no network, hashing or filesystem work in those critical sections. A fill
runs in its initiating caller, so cancellation joins every byte operation before
that allocation releases; followers can retry with their own grant if it stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import ssl
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Literal, Protocol

from contractor_runtime.archive_cache_files import (
    MARKER,
    MAX_ENTRY_METADATA_BYTES,
    CacheFiles,
)
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactMetadataResult, ArtifactRef
from contractor_runtime.shared_config import artifact_payload_max_bytes
from contractor_runtime.threads import consume_task_exception, to_thread_until_done

MAX_CACHE_BYTES = 1 << 30
MAX_CACHE_ENTRIES = 1024
MAX_CACHE_METADATA_BYTES = 4 << 20
MAX_CACHE_FILLS = 4
MAX_CACHE_WAITERS = 64
CACHE_TTL_SECONDS = 24 * 60 * 60
MAINTENANCE_SECONDS = 60
MAINTENANCE_ENTRIES = 64
logger = logging.getLogger(__name__)


class ArchiveReader(Protocol):
    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue: ...


@dataclass(frozen=True, slots=True)
class ArchiveValue:
    artifact: ArtifactRef
    media_type: str
    data: bytes = field(repr=False)


async def read_archive(
    reader: ArchiveReader, ref: ArtifactRef, *, kind: Literal["source", "skill"] = "source"
) -> ArtifactValue | ArchiveValue:
    """Opt in only from workspace source and selected Skill package preparation."""

    if isinstance(reader, ArtifactClient) and reader.archive_cache is not None:
        return await reader.archive_cache.read(reader, ref, kind=kind)
    return await reader.read_artifact(ref)


def tls_trust_fingerprint(context: ssl.SSLContext) -> bytes:
    """Bind reuse to the effective CA set and TLS verification configuration."""

    digest = hashlib.sha256()
    for certificate in sorted(context.get_ca_certs(binary_form=True)):
        digest.update(len(certificate).to_bytes(4, "big"))
        digest.update(certificate)
    digest.update(f"{context.verify_mode}:{context.verify_flags}:{context.check_hostname}".encode())
    return digest.digest()


def archive_trust_domain(endpoint: str, tls_fingerprint: bytes) -> str:
    return hashlib.sha256(endpoint.encode() + b"\0" + tls_fingerprint).hexdigest()


@dataclass(slots=True)
class _Entry:
    stem: str
    size: int
    media_type: str
    digest: str
    metadata: bytes = field(repr=False)
    downloaded: float
    readers: int = 0
    retired: bool = False
    removing: bool = False


class ImmutableArchiveCache:
    def __init__(
        self,
        files: CacheFiles,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_bytes: int = MAX_CACHE_BYTES,
        max_entries: int = MAX_CACHE_ENTRIES,
        max_metadata_bytes: int = MAX_CACHE_METADATA_BYTES,
    ) -> None:
        self._files = files
        self._clock = clock
        self._max_bytes = min(max_bytes, MAX_CACHE_BYTES)
        self._max_entries = min(max_entries, MAX_CACHE_ENTRIES)
        self._max_metadata_bytes = min(max_metadata_bytes, MAX_CACHE_METADATA_BYTES)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._expiry: OrderedDict[str, _Entry] = OrderedDict()
        self._fills: dict[str, asyncio.Future[None]] = {}
        self._waiters = 0
        self._slots = 0
        self._charged = len(MARKER)
        self._metadata = len(MARKER)
        self._disabled = False
        self._closing = False
        self._users = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._maintenance: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._counts = dict.fromkeys(
            ("hit", "miss", "bypass", "corruption", "sourceHit", "skillHit"), 0
        )

    @classmethod
    async def open(cls, root: Path) -> ImmutableArchiveCache | None:
        files = CacheFiles(root)
        try:
            ready = await to_thread_until_done(files.initialize, name="archive-cache-startup")
        except asyncio.CancelledError:
            files.close()
            raise
        if not ready:
            logger.info("runtime archive cache disabled after unsafe or unavailable local state")
            return None
        cache = cls(files)
        cache._maintenance = asyncio.create_task(cache._maintain(), name="archive-cache-expiry")
        return cache

    @property
    def counters(self) -> dict[str, int]:
        return dict(
            self._counts,
            chargedBytes=self._charged,
            metadataBytes=self._metadata,
            entries=self._slots,
            fills=len(self._fills),
            waiters=self._waiters,
        )

    async def read(
        self, client: ArtifactClient, ref: ArtifactRef, *, kind: Literal["source", "skill"]
    ) -> ArchiveValue:
        self._users += 1
        self._idle.clear()
        try:
            # This must succeed even for exact refs, hits, disabled population,
            # coalesced readers and entries whose previous owner was deleted.
            metadata = await client.read_artifact_metadata(ref)
            if not client.archive_trust_domain:
                raise ArtifactTransportError("archive cache trust domain is missing")
            stem = hashlib.sha256(
                (client.archive_trust_domain + "\0" + metadata.content_cache_key).encode("ascii")
            ).hexdigest()
            return await self._authorized_read(client, metadata, stem, kind)
        finally:
            self._users -= 1
            if not self._users:
                self._idle.set()

    async def _authorized_read(
        self,
        client: ArtifactClient,
        metadata: ArtifactMetadataResult,
        stem: str,
        kind: Literal["source", "skill"],
    ) -> ArchiveValue:
        if not self._closing and not self._disabled:
            entry = self._entries.get(stem)
            if entry is not None:
                value = await self._cached(entry, metadata)
                if value is not None:
                    client.observe_cached_read(metadata.artifact)
                    self._counts["hit"] += 1
                    self._counts[kind + "Hit"] += 1
                    return value
        future = self._fills.get(stem)
        if future is not None and self._waiters < MAX_CACHE_WAITERS:
            self._waiters += 1
            try:
                await asyncio.shield(future)
            finally:
                self._waiters -= 1
            entry = self._entries.get(stem)
            if entry is not None and not self._disabled and not self._closing:
                value = await self._cached(entry, metadata)
                if value is not None:
                    client.observe_cached_read(metadata.artifact)
                    self._counts["hit"] += 1
                    self._counts[kind + "Hit"] += 1
                    return value
            # The original owner may have cancelled or storage failed. The
            # follower fetches on its own grant; never inherit another request.
        self._counts["miss"] += 1
        if (
            stem not in self._fills
            and stem not in self._entries
            and len(self._fills) < MAX_CACHE_FILLS
            and not self._disabled
            and not self._closing
        ):
            future = asyncio.get_running_loop().create_future()
            self._fills[stem] = future
            try:
                return await self._populate(client, metadata, stem)
            finally:
                self._fills.pop(stem)
                future.set_result(None)
        self._counts["bypass"] += 1
        return await self._fetch(client, metadata)

    async def _populate(
        self, client: ArtifactClient, metadata: ArtifactMetadataResult, stem: str
    ) -> ArchiveValue:
        reserved = await self._reserve(metadata.size)
        if not reserved:
            self._counts["bypass"] += 1
            return await self._fetch(client, metadata)
        promoted = False
        try:
            value = await self._fetch(client, metadata)
            downloaded = self._clock()
            digest, encoded = await to_thread_until_done(
                _integrity_metadata, value.data, value.media_type, name="archive-cache-integrity"
            )
            try:
                await to_thread_until_done(
                    self._files.write, stem, value.data, encoded, name="archive-cache-write"
                )
            except OSError:
                return value
            entry = _Entry(stem, metadata.size, metadata.media_type, digest, encoded, downloaded)
            self._entries[stem] = entry
            self._expiry[stem] = entry
            # Atomically convert the same reservation after rename; no extra
            # file charge, even if cancellation arrived during the write.
            refund = MAX_ENTRY_METADATA_BYTES - len(encoded)
            self._charged -= refund
            self._metadata -= refund
            promoted = True
            return value
        finally:
            if not promoted:
                removed = False
                try:
                    removed = await to_thread_until_done(
                        self._files.remove, stem, name="archive-cache-fill-cleanup"
                    )
                finally:
                    if removed:
                        self._charged -= metadata.size + MAX_ENTRY_METADATA_BYTES
                        self._metadata -= MAX_ENTRY_METADATA_BYTES
                        self._slots -= 1
                    else:
                        self._disabled = True  # Keep the entire reservation charged.

    async def _fetch(
        self, client: ArtifactClient, metadata: ArtifactMetadataResult
    ) -> ArchiveValue:
        value = await client.read_artifact(metadata.artifact)
        if (
            value.artifact != metadata.artifact
            or value.media_type != metadata.media_type
            or len(value.data) != metadata.size
        ):
            raise ArtifactTransportError("archive payload does not match authorized metadata")
        return ArchiveValue(value.artifact, value.media_type, value.data)

    async def _cached(self, entry: _Entry, metadata: ArtifactMetadataResult) -> ArchiveValue | None:
        if entry.retired or entry.removing:
            return None
        if self._clock() - entry.downloaded >= CACHE_TTL_SECONDS:
            entry.retired = True
            if not entry.readers:
                await self._remove(entry)
            return None
        if entry.size != metadata.size or entry.media_type != metadata.media_type:
            # Valid metadata that disagrees with a local entry never licenses
            # returning it. Discard and fetch the exact authorized version.
            entry.retired = True
            self._counts["corruption"] += 1
            if not entry.readers:
                await self._remove(entry)
            return None
        entry.readers += 1
        self._entries.move_to_end(entry.stem)
        try:
            try:
                data = await to_thread_until_done(
                    self._files.read,
                    entry.stem,
                    entry.size,
                    entry.metadata,
                    entry.digest,
                    name="archive-cache-read",
                )
            except OSError:
                self._counts["corruption"] += 1
                entry.retired = True
                return None
            return ArchiveValue(metadata.artifact, metadata.media_type, data)
        finally:
            entry.readers -= 1
            if entry.retired and not entry.readers:
                await self._remove(entry)

    async def _reserve(self, size: int) -> bool:
        if not 0 <= size <= artifact_payload_max_bytes():
            return False
        # One admission pass examines at most 64 LRU candidates. No unbounded
        # queue, whole-index scan or network/file work inside bookkeeping.
        candidates = iter(list(islice(self._entries.values(), MAINTENANCE_ENTRIES)))
        while not self._disabled and not self._closing:
            if (
                self._charged + size + MAX_ENTRY_METADATA_BYTES <= self._max_bytes
                and self._metadata + MAX_ENTRY_METADATA_BYTES <= self._max_metadata_bytes
                and self._slots < self._max_entries
            ):
                self._charged += size + MAX_ENTRY_METADATA_BYTES
                self._metadata += MAX_ENTRY_METADATA_BYTES
                self._slots += 1
                return True
            entry = next(candidates, None)
            if entry is None:
                return False
            if not entry.readers and not entry.removing:
                await self._remove(entry)
        return False

    async def _remove(self, entry: _Entry) -> None:
        if entry.readers or entry.removing or self._entries.get(entry.stem) is not entry:
            return
        entry.retired = True
        entry.removing = True
        removed = False
        try:
            removed = await to_thread_until_done(
                self._files.remove, entry.stem, name="archive-cache-eviction"
            )
        finally:
            entry.removing = False
            if removed:
                self._entries.pop(entry.stem, None)
                self._expiry.pop(entry.stem, None)
                self._charged -= entry.size + len(entry.metadata)
                self._metadata -= len(entry.metadata)
                self._slots -= 1
            else:
                self._disabled = True

    async def expire(self) -> None:
        for entry in list(islice(self._expiry.values(), MAINTENANCE_ENTRIES)):
            if self._expiry.get(entry.stem) is not entry:
                continue
            # Rotate even busy entries: one active oldest reader cannot starve
            # expiry of later entries. Hit recency never moves this cursor.
            self._expiry.move_to_end(entry.stem)
            if self._clock() - entry.downloaded >= CACHE_TTL_SECONDS:
                entry.retired = True
                if not entry.readers:
                    await self._remove(entry)

    async def _maintain(self) -> None:
        next_pass = asyncio.get_running_loop().time()
        while True:
            next_pass += MAINTENANCE_SECONDS
            await asyncio.sleep(max(0.0, next_pass - asyncio.get_running_loop().time()))
            await self.expire()

    async def close(self) -> None:
        # Like to_thread_until_done, cancellation cannot release the work-root
        # lock while an uninterruptible reader still owns files. The Runtime's
        # existing stop/fence deadline owns inability to join, never a detached
        # cache task. Concurrent/repeated close calls join the same finalizer.
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close(), name="archive-cache-shutdown")
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.wait({self._close_task})
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            consume_task_exception(self._close_task)
            raise asyncio.CancelledError
        self._close_task.result()

    async def _close(self) -> None:
        if self._maintenance is not None:
            self._maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance
        await self._idle.wait()
        # Shutdown may remove the bounded complete index; regular passes have
        # the stricter 64-entry work bound. The outer Runtime deadline owns it.
        for entry in tuple(self._entries.values()):
            await self._remove(entry)
        logger.info(
            "runtime archive cache stopped",
            extra={
                "archiveCacheHit": self._counts["hit"],
                "archiveCacheMiss": self._counts["miss"],
                "archiveCacheBypass": self._counts["bypass"],
                "archiveCacheCorruption": self._counts["corruption"],
                "archiveCacheSourceHit": self._counts["sourceHit"],
                "archiveCacheSkillHit": self._counts["skillHit"],
                "archiveCacheChargedBytes": self._charged,
            },
        )
        await to_thread_until_done(self._files.close, name="archive-cache-close")


def _integrity_metadata(data: bytes, media_type: str) -> tuple[str, bytes]:
    digest = hashlib.sha256(data).hexdigest()
    metadata = json.dumps(
        {"sha256": digest, "size": len(data), "mediaType": media_type}, separators=(",", ":")
    ).encode()
    return digest, metadata
