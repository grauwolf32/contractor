from __future__ import annotations

import asyncio
import errno
import json
import os
import ssl
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fakes.allocation_rpc import AllocationRPCFixture
from shared_config_defaults import MAX_ARTIFACT_BYTES
from test_artifacts import TIMESTAMP_HEADERS

from contractor_runtime.archive_cache import (
    CACHE_TTL_SECONDS,
    MAX_CACHE_BYTES,
    MAX_CACHE_ENTRIES,
    MAX_CACHE_FILLS,
    MAX_CACHE_METADATA_BYTES,
    MAX_CACHE_WAITERS,
    ImmutableArchiveCache,
    archive_trust_domain,
    read_archive,
    tls_trust_fingerprint,
)
from contractor_runtime.archive_cache_files import CACHE_DIRECTORY, MARKER, CacheFiles
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactHTTPResponse,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef


class ArchiveTransport(AllocationRPCFixture):
    """Same immutable version, independently authorized allocation binding refs."""

    def __init__(self, payload: bytes = b"archive", media_type: str = "application/zip") -> None:
        self.payload = payload
        self.media_type = media_type
        self.key = "version_immutable"
        self.metadata_calls = 0
        self.payload_calls = 0
        self.denied = 0
        self.invalid: dict[str, object] = {}
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.download_error: Exception | None = None

    async def request(self, method, path, *, headers, body, max_response_bytes):
        del method, headers, body, max_response_bytes
        parsed = urlsplit(path)
        allocation = parsed.path.split("/")[2]
        revision = parse_qs(parsed.query).get("revision", [f"binding-{allocation}"])[0]
        if parsed.path.endswith("/metadata"):
            self.metadata_calls += 1
            if self.denied:
                if self.denied == 599:
                    raise ArtifactTransportError("metadata response lost")
                return ArtifactHTTPResponse(
                    self.denied,
                    {"content-type": "application/json"},
                    json.dumps(
                        {
                            "code": "artifact_not_found"
                            if self.denied == 404
                            else "artifact_access_denied",
                            "message": "unavailable",
                            "retryable": False,
                            "requestId": "test",
                        }
                    ).encode(),
                )
            return ArtifactHTTPResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "apiVersion": "contractor/v1alpha1",
                        "artifact": {
                            "namespace": parsed.path.split("/")[-3],
                            "name": parsed.path.split("/")[-2],
                            "revision": revision,
                        },
                        "mediaType": self.media_type,
                        "size": len(self.payload),
                        "contentCacheKey": self.key,
                        **self.invalid,
                    }
                ).encode(),
            )
        self.payload_calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.download_error is not None:
            raise self.download_error
        return ArtifactHTTPResponse(
            200,
            {"content-type": self.media_type, "etag": f'"{revision}"', **TIMESTAMP_HEADERS},
            self.payload,
        )


def client(cache, transport, allocation="one", domain="server-a"):
    return ArtifactClient(allocation, transport, archive_cache=cache, archive_trust_domain=domain)


def source(revision=None):
    return ArtifactRef(
        namespace="inputs", name="source", **({"revision": revision} if revision else {})
    )


def authorized_read_barrier(cache, expected):
    """Observe entry after metadata validation, independent of thread scheduling."""
    ready = asyncio.Event()
    original = cache._authorized_read
    entered = 0

    async def observed(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == expected:
            ready.set()
        return await original(*args, **kwargs)

    cache._authorized_read = observed
    return ready


def make_cache(tmp_path, **limits):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = CacheFiles(tmp_path)
    assert files.initialize()
    return ImmutableArchiveCache(files, **limits)


def test_authorized_forks_reuse_bytes_with_consuming_provenance(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        first, second = client(cache, transport), client(cache, transport, "two")
        assert (await read_archive(first, source())).artifact.revision == "binding-one"
        assert (await read_archive(second, source())).artifact.revision == "binding-two"
        assert transport.metadata_calls == 2 and transport.payload_calls == 1
        assert [ref.revision for ref in first.known_exact_refs] == ["binding-one"]
        assert [ref.revision for ref in second.observed_exact_refs_since(0)] == ["binding-two"]
        assert cache.counters["hit"] == 1
        # Different versions and Server trust domains never alias equal bytes.
        await read_archive(client(cache, transport, "three", "server-b"), source())
        transport.key = "independent-version"
        await read_archive(second, source("new-current"))
        assert transport.payload_calls == 3
        await cache.close()
        assert list((tmp_path / CACHE_DIRECTORY).iterdir()) == [
            tmp_path / CACHE_DIRECTORY / ".owner"
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("denied", [403, 404, 599])
def test_exact_hits_cannot_bypass_revocation_purge_or_lost_metadata(tmp_path, denied):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        reader = client(cache, transport)
        await read_archive(reader, source("exact"))
        transport.denied = denied
        with pytest.raises((ArtifactAPIError, ArtifactTransportError)):
            await read_archive(reader, source("exact"))
        assert transport.metadata_calls == 2 and transport.payload_calls == 1
        assert reader.observation_cursor == 1
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "invalid",
    [
        {"contentCacheKey": "../bad"},
        {"contentCacheKey": ""},
        {"contentCacheKey": None},
        {"size": MAX_ARTIFACT_BYTES + 1},
        {"extra": 1},
    ],
)
def test_invalid_metadata_is_transport_failure_without_raw_fetch(tmp_path, invalid):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        transport.invalid = invalid
        with pytest.raises(ArtifactTransportError):
            await read_archive(client(cache, transport), source())
        assert transport.payload_calls == 0
        await cache.close()

    asyncio.run(scenario())


def test_same_key_coalescing_and_cancelled_leader_use_followers_own_grant(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        transport.gate = asyncio.Event()
        leader = asyncio.create_task(read_archive(client(cache, transport), source()))
        await transport.started.wait()
        ready = authorized_read_barrier(cache, 4)
        followers = [
            asyncio.create_task(read_archive(client(cache, transport, f"f-{i}"), source()))
            for i in range(4)
        ]
        await asyncio.wait_for(ready.wait(), 5)
        assert cache.counters["waiters"] == 4
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        transport.gate.set()
        results = await asyncio.gather(*followers)
        assert [v.artifact.revision for v in results] == [f"binding-f-{i}" for i in range(4)]
        assert transport.payload_calls == 5
        assert cache.counters["waiters"] == cache.counters["fills"] == 0
        assert cache.counters["entries"] == 1
        await cache.close()

    asyncio.run(scenario())


def test_distinct_fill_and_waiter_caps_bypass_without_growing_queue(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        transports = [ArchiveTransport() for _ in range(5)]
        for i, transport in enumerate(transports):
            transport.key = f"key-{i}"
            transport.gate = asyncio.Event()
        leaders = [
            asyncio.create_task(read_archive(client(cache, t, f"l-{i}"), source()))
            for i, t in enumerate(transports)
        ]
        await asyncio.gather(*(t.started.wait() for t in transports))
        assert cache.counters["fills"] == MAX_CACHE_FILLS
        assert cache.counters["entries"] == 4
        ready = authorized_read_barrier(cache, MAX_CACHE_WAITERS + 1)
        followers = [
            asyncio.create_task(read_archive(client(cache, transports[0], f"w-{i}"), source()))
            for i in range(MAX_CACHE_WAITERS + 1)
        ]
        await asyncio.wait_for(ready.wait(), 5)
        assert cache.counters["waiters"] == MAX_CACHE_WAITERS
        assert transports[0].payload_calls == 2  # 65th waiter bypasses.
        assert cache.counters["bypass"] == 2  # Fifth distinct key also bypasses.
        for transport in transports:
            transport.gate.set()
        await asyncio.gather(*leaders, *followers)
        assert cache.counters["waiters"] == 0
        assert cache.counters["entries"] == 4
        await cache.close()

    asyncio.run(scenario())


def test_actual_gib_reservations_and_metadata_entry_limits_without_large_files(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        # Full 64-MiB reservations include metadata, so the sixteenth cannot fit.
        for _ in range(15):
            assert await cache._reserve(MAX_ARTIFACT_BYTES)
        assert not await cache._reserve(MAX_ARTIFACT_BYTES)
        remaining = MAX_CACHE_BYTES - cache.counters["chargedBytes"] - 4096
        assert await cache._reserve(remaining)
        assert cache.counters["chargedBytes"] == MAX_CACHE_BYTES
        assert not await cache._reserve(0)
        cache._files.close()
        cache = make_cache(tmp_path)
        for _ in range(MAX_CACHE_ENTRIES - 1):
            assert await cache._reserve(0)
        # Marker is charged too: the metadata ceiling is reached before a
        # 1024th full metadata reservation. Completed compact entries fit 1024.
        assert not await cache._reserve(0)
        assert cache.counters["metadataBytes"] <= MAX_CACHE_METADATA_BYTES
        cache._files.close()

    asyncio.run(scenario())


def test_lru_eviction_and_in_use_entry_bypass_real_files(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path, max_entries=2)
        transport = ArchiveTransport()
        reader = client(cache, transport)
        for key in ("a", "b", "a", "c"):
            transport.key = key
            await read_archive(reader, source())
        assert transport.payload_calls == 3
        # 'a' was touched; 'b' is the LRU victim.
        transport.key = "a"
        await read_archive(reader, source())
        assert transport.payload_calls == 3
        entries = list(cache._entries.values())
        for entry in entries:
            entry.readers += 1
        transport.key = "d"
        await read_archive(reader, source())
        assert cache.counters["bypass"] == 1 and cache.counters["entries"] == 2
        for entry in entries:
            entry.readers -= 1
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["corrupt", "truncate", "metadata", "symlink", "hardlink"])
def test_invalid_files_never_return_bytes_or_follow_links(tmp_path, fault):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        reader = client(cache, transport)
        await read_archive(reader, source())
        entry = next(iter(cache._entries.values()))
        path = cache._files.path / (entry.stem + ".data")
        external = tmp_path / "external"
        external.write_bytes(b"outside")
        if fault == "metadata":
            path.with_suffix(".meta").write_bytes(b"invalid")
        elif fault in {"hardlink", "symlink"}:
            path.unlink()
            if fault == "hardlink":
                os.link(external, path)
            else:
                path.symlink_to(external)
        else:
            path.write_bytes(b"bad" if fault == "truncate" else b"corrupt")
        assert (await read_archive(reader, source())).data == transport.payload
        assert transport.payload_calls == 2 and cache.counters["corruption"] == 1
        assert external.read_bytes() == b"outside"
        await cache.close()

    asyncio.run(scenario())


def test_enospc_bypass_and_failed_cleanup_keeps_full_charge(tmp_path, monkeypatch):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()

        def no_space(*args):
            raise OSError(errno.ENOSPC, "no space")

        monkeypatch.setattr(cache._files, "write", no_space)
        assert (await read_archive(client(cache, transport), source())).data == b"archive"
        assert cache.counters["entries"] == 0
        monkeypatch.setattr(cache._files, "remove", lambda _: False)
        assert (await read_archive(client(cache, transport), source())).data == b"archive"
        assert cache.counters["chargedBytes"] == len(MARKER) + len(b"archive") + 4096
        assert cache._disabled
        before = cache.counters["chargedBytes"]
        await read_archive(client(cache, transport), source())
        assert cache.counters["chargedBytes"] == before
        await cache.close()

    asyncio.run(scenario())


def test_cancellation_during_atomic_write_joins_before_releasing_reservation(tmp_path, monkeypatch):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        started, finish = threading.Event(), threading.Event()
        original = cache._files.write

        def blocked(*args):
            started.set()
            finish.wait()
            original(*args)

        monkeypatch.setattr(cache._files, "write", blocked)
        task = asyncio.create_task(read_archive(client(cache, transport), source()))
        await asyncio.to_thread(started.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and cache.counters["entries"] == 1
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cache.counters["entries"] == 0 and cache.counters["chargedBytes"] == len(MARKER)
        assert {p.name for p in cache._files.path.iterdir()} == {".owner"}
        await cache.close()

    asyncio.run(scenario())


def test_non_sliding_ttl_and_bounded_expiry_pass(tmp_path):
    async def scenario():
        now = [0.0]
        cache = make_cache(tmp_path, clock=lambda: now[0])
        transport = ArchiveTransport()
        reader = client(cache, transport)
        for i in range(130):
            transport.key = f"version-{i}"
            await read_archive(reader, source())
        now[0] = CACHE_TTL_SECONDS - 1
        await read_archive(reader, source())
        assert cache.counters["hit"] == 1
        now[0] += 1
        oldest = next(iter(cache._entries.values()))
        oldest.readers = 1
        await cache.expire()
        assert cache.counters["entries"] == 67  # 64 examined, 63 removed.
        assert oldest.retired
        transport.key = "version-0"
        await read_archive(reader, source())
        assert transport.payload_calls == 131  # Busy expired entry cannot serve a hit.
        oldest.readers = 0
        await cache.expire()
        await cache.expire()
        assert cache.counters["entries"] == 0
        await cache.close()

    asyncio.run(scenario())


def test_restart_removes_only_marker_owned_files_and_never_reuses_entries(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        await read_archive(client(cache, transport), source())
        # Simulate a process dying without graceful cleanup.
        cache._files.close()
        workspace = tmp_path / "workspace-live"
        workspace.mkdir()
        (workspace / "keep").write_bytes(b"live")
        reopened = make_cache(tmp_path)
        await read_archive(client(reopened, transport), source())
        assert transport.payload_calls == 2
        assert (workspace / "keep").read_bytes() == b"live"
        await reopened.close()
        unknown = tmp_path / CACHE_DIRECTORY / "operator-note"
        unknown.write_bytes(b"keep")
        assert await ImmutableArchiveCache.open(tmp_path) is None
        assert unknown.read_bytes() == b"keep"

    asyncio.run(scenario())


def test_unowned_or_symlink_root_disables_cache_without_deletion(tmp_path):
    cache_root = tmp_path / CACHE_DIRECTORY
    external = tmp_path / "external"
    external.mkdir()
    cache_root.symlink_to(external, target_is_directory=True)
    assert not CacheFiles(tmp_path).initialize()
    assert cache_root.is_symlink() and external.is_dir()
    cache_root.unlink()
    cache_root.mkdir(mode=0o700)
    (cache_root / "unowned").write_bytes(b"keep")
    assert not CacheFiles(tmp_path).initialize()
    assert (cache_root / "unowned").read_bytes() == b"keep"


def test_trust_domain_binds_endpoint_and_effective_tls_trust():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    first = tls_trust_fingerprint(context)
    context.check_hostname = False
    second = tls_trust_fingerprint(context)
    assert first != second
    assert archive_trust_domain("https://server/private/v1", first) != archive_trust_domain(
        "https://other/private/v1", first
    )
    assert archive_trust_domain("https://server/private/v1", first) != archive_trust_domain(
        "https://server/private/v1", second
    )


def test_same_key_success_downloads_once(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        transport.gate = asyncio.Event()
        leader = asyncio.create_task(read_archive(client(cache, transport), source()))
        await transport.started.wait()
        ready = authorized_read_barrier(cache, 1)
        follower = asyncio.create_task(read_archive(client(cache, transport, "fork"), source()))
        await asyncio.wait_for(ready.wait(), 5)
        assert cache.counters["waiters"] == 1
        transport.gate.set()
        await asyncio.gather(leader, follower)
        assert transport.payload_calls == 1 and transport.metadata_calls == 2
        await cache.close()

    asyncio.run(scenario())


def test_production_workspace_hydration_reuses_archive_but_snapshots_fresh_disk(tmp_path):
    from test_projectfs_zip import archive, settings

    from contractor_runtime.contracts import AllocationWorkspaceSource, AllocationWorkspaceSpec
    from contractor_runtime.projectfs import LocalWorkspaceProvider, hydrate_workspace

    async def scenario():
        root = tmp_path / "workspaces"
        cache = make_cache(root)
        transport = ArchiveTransport(archive({"source.txt": b"fresh\n"}))
        provider = LocalWorkspaceProvider(settings("local", root))
        roots = []
        for allocation in ("one", "two"):
            reader = client(cache, transport, allocation)
            exact = source(f"binding-{allocation}")
            spec = AllocationWorkspaceSpec(
                mode="direct", sources=[AllocationWorkspaceSource(artifact=exact, target="")]
            )
            workspace = await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id=allocation,
                timeout_seconds=5,
            )
            path = Path(workspace.storage.root) / "run_workdir" / "source.txt"
            roots.append(path.parent)
            assert path.read_bytes() == b"fresh\n"
            path.write_bytes(b"local\n")
            assert (await workspace.snapshot()).files[0].text == "local\n"
            assert reader.observed_exact_refs_since(0) == (exact,)
            await workspace.close()
            await provider.cleanup(workspace.storage)
            assert not path.exists()
        assert roots[0] != roots[1]
        assert transport.metadata_calls == 2 and transport.payload_calls == 1
        assert cache.counters["entries"] == 1
        await cache.close()

    asyncio.run(scenario())


def test_production_skill_hits_revalidate_trusted_digest_and_manifest(tmp_path):
    from test_agent_skill_toolset import resolved_value, runtime_settings, skill_package

    from contractor_runtime.agent_skills import MEDIA_TYPE
    from contractor_runtime.agent_skills.runtime import (
        AgentSkillPreparationError,
        prepare_agent_skills,
    )
    from contractor_runtime.workspace import AllocationWorkspace

    async def scenario():
        cache = make_cache(tmp_path / "cache-root")
        payload = skill_package()
        selected, _ = resolved_value(payload)
        transport = ArchiveTransport(payload, MEDIA_TYPE)
        roots = []
        for allocation in ("one", "two", "bad-digest", "bad-manifest"):
            path = tmp_path / allocation
            path.mkdir()
            reader = client(cache, transport, allocation)
            roots.append(path)
            picked = selected.model_copy(
                update={
                    "artifact": ArtifactRef(
                        namespace="skills", name="demo", revision=f"skill-{allocation}"
                    )
                }
            )
            if allocation == "bad-digest":
                picked = picked.model_copy(update={"package_digest": "sha256:" + "0" * 64})
            if allocation == "bad-manifest":
                picked = picked.model_copy(update={"name": "other"})

            async def prepare(picked=picked, allocation=allocation, path=path, reader=reader):
                return await prepare_agent_skills(
                    [picked],
                    allocation_id=allocation,
                    runtime_settings=runtime_settings(),
                    workspace=AllocationWorkspace(root=path.parent, path=path),
                    artifact_client_factory=lambda *_, reader=reader: reader,
                )

            if allocation.startswith("bad"):
                with pytest.raises(AgentSkillPreparationError) as caught:
                    await prepare()
                assert not caught.value.retryable
            else:
                owner = await prepare()
                assert owner is not None and owner.skills[0].name == "demo"
                assert reader.observed_exact_refs_since(0) == (picked.artifact,)
                await owner.close()
                assert not owner.extraction_root.exists()
        assert transport.metadata_calls == 4 and transport.payload_calls == 1
        await cache.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("storage", ["local", "memory", "disabled"])
def test_actual_cli_startup_gates_cache_by_workspace_storage(tmp_path, monkeypatch, storage):
    import dataclasses

    from test_cli import ConfigTransport, make_settings
    from test_projectfs_zip import settings

    import contractor_runtime.cli as cli

    async def scenario():
        workspace_settings = (
            settings(storage, tmp_path / "project")
            if storage == "local"
            else (settings("memory") if storage == "memory" else None)
        )
        configured = dataclasses.replace(make_settings(tmp_path), workspace=workspace_settings)

        async def serve_stub(*args, archive_cache, **kwargs):
            if storage == "local":
                assert archive_cache is not None
                transport = ArchiveTransport()
                await read_archive(client(archive_cache, transport), source())
                assert list((tmp_path / "project" / CACHE_DIRECTORY).glob("*.data"))
            else:
                assert archive_cache is None
                assert not list(tmp_path.rglob(CACHE_DIRECTORY))

        monkeypatch.setattr(cli, "_serve", serve_stub)
        await cli.serve(configured, transport=ConfigTransport())
        assert not list(tmp_path.rglob("*.data"))

    asyncio.run(scenario())


def test_completed_index_at_1024_entries_and_parallel_reservations_stay_bounded(tmp_path):
    async def scenario():
        cache = make_cache(tmp_path, max_bytes=256 << 10)
        transport = ArchiveTransport(b"small")
        reader = client(cache, transport)
        for i in range(MAX_CACHE_ENTRIES):
            transport.key = f"version-{i}"
            await read_archive(reader, source())
        assert cache.counters["entries"] == MAX_CACHE_ENTRIES
        assert len(list(cache._files.path.glob("*.data"))) == MAX_CACHE_ENTRIES
        assert cache.counters["chargedBytes"] <= 256 << 10
        # Four concurrent distinct-key admissions must account for each
        # reservation before any bytes arrive, even while LRU eviction awaits I/O.
        entries = list(cache._entries.values())
        for entry in entries:
            entry.readers += 1
        pending = []
        transports = []
        for i in range(4):
            other = ArchiveTransport()
            other.key = f"new-{i}"
            other.gate = asyncio.Event()
            transports.append(other)
            pending.append(
                asyncio.create_task(read_archive(client(cache, other, f"a-{i}"), source()))
            )
        await asyncio.gather(*(t.started.wait() for t in transports))
        assert cache.counters["entries"] == MAX_CACHE_ENTRIES
        assert cache.counters["fills"] <= MAX_CACHE_FILLS
        assert cache.counters["bypass"] == 4
        for transport in transports:
            transport.gate.set()
        await asyncio.gather(*pending)
        for entry in entries:
            entry.readers -= 1
        await cache.close()

    asyncio.run(scenario())


def test_active_reader_finishes_after_ttl_but_new_reader_fetches(tmp_path, monkeypatch):
    async def scenario():
        now = [0.0]
        cache = make_cache(tmp_path, clock=lambda: now[0])
        transport = ArchiveTransport()
        reader = client(cache, transport)
        await read_archive(reader, source())
        started, finish = threading.Event(), threading.Event()
        original = cache._files.read

        def blocked(*args):
            started.set()
            finish.wait()
            return original(*args)

        monkeypatch.setattr(cache._files, "read", blocked)
        active = asyncio.create_task(read_archive(reader, source()))
        await asyncio.to_thread(started.wait)
        now[0] = CACHE_TTL_SECONDS
        await cache.expire()
        assert cache.counters["entries"] == 1
        assert (await read_archive(client(cache, transport, "new"), source())).data == b"archive"
        assert transport.payload_calls == 2
        assert cache.counters["entries"] == 1
        close = asyncio.create_task(cache.close())
        await asyncio.sleep(0)
        assert not close.done()
        finish.set()
        assert (await active).data == b"archive"
        await close
        assert cache.counters["entries"] == 0

    asyncio.run(scenario())


def test_cancelled_close_joins_reader_before_releasing_files(tmp_path, monkeypatch):
    async def scenario():
        cache = make_cache(tmp_path)
        transport = ArchiveTransport()
        reader = client(cache, transport)
        await read_archive(reader, source())
        started, finish = threading.Event(), threading.Event()
        original = cache._files.read

        def blocked(*args):
            started.set()
            finish.wait()
            return original(*args)

        monkeypatch.setattr(cache._files, "read", blocked)
        active = asyncio.create_task(read_archive(reader, source()))
        await asyncio.to_thread(started.wait)
        closing = asyncio.create_task(cache.close())
        await asyncio.sleep(0)
        closing.cancel()
        active.cancel()
        await asyncio.sleep(0)
        assert not active.done() and not closing.done()
        assert cache._files.fd >= 0
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert cache._files.fd == -1 and cache.counters["entries"] == 0
        await cache.close()  # Repeated close joins the same completed finalizer.

    asyncio.run(scenario())


def test_parallel_fills_reserve_full_bytes_before_any_write(tmp_path):
    async def scenario():
        payload = b"x" * 12000
        maximum = len(MARKER) + 2 * (len(payload) + 4096)
        cache = make_cache(tmp_path, max_bytes=maximum)
        transports, pending = [], []
        for i in range(4):
            transport = ArchiveTransport(payload)
            transport.key = f"version-{i}"
            transport.gate = asyncio.Event()
            transports.append(transport)
            pending.append(
                asyncio.create_task(read_archive(client(cache, transport, f"a-{i}"), source()))
            )
        await asyncio.gather(*(t.started.wait() for t in transports))
        assert cache.counters["chargedBytes"] == maximum
        assert cache.counters["entries"] == 2 and cache.counters["bypass"] == 2
        for transport in transports:
            transport.gate.set()
        await asyncio.gather(*pending)
        assert cache.counters["chargedBytes"] <= maximum
        actual_bytes = sum(path.stat().st_size for path in cache._files.path.iterdir())
        assert cache.counters["chargedBytes"] == actual_bytes
        await cache.close()

    asyncio.run(scenario())
