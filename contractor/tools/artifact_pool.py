"""Cross-namespace read/search surface over the whole artifact pool.

``memory_tools(name=namespace)`` is scoped to a *single* memory namespace (the
current task's) and never sees task-result artifacts. ``artifact_pool`` is the
read-only complement: one tool surface that lists, reads, and searches across
*every* namespace and artifact kind in the run.

Why this works in one call: ``save_result_artifacts`` and ``MemoryTools`` both
persist with ``session_id=None``, and ADK's ``FileArtifactService`` treats
``session_id is None`` as user-scoped — so the entire pool lives under one flat
keyspace. ``list_artifact_keys(app_name, user_id, session_id=None)`` enumerates
all of it; ``load_artifact(filename, session_id=None)`` reads any entry.

The keyspace has exactly three shapes::

    user:memory/<namespace>                       # a MemoryNote store (YAML)
    <artifact_key>/{result|summary|records}       # task outputs
    <bespoke-key>                                  # workflow raw artifacts

Search is delegated to a pluggable :class:`ArtifactPoolBackend`. The default
:class:`KeywordPoolBackend` is a dependency-free term-frequency ranker; a future
embedding / pgvector backend drops in behind the same protocol without touching
the frontend tools — see ``search`` and the module footer for the RAG seam.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import yaml
from google.adk.artifacts import BaseArtifactService

from contractor.tools.result import aguard, err, ok, ok_page
from contractor.utils.settings import get_settings

# ─── Keyspace model ───────────────────────────────────────────────────────────

PoolKind = Literal["memory", "result", "summary", "records", "raw"]

_MEMORY_PREFIX = "user:memory/"
_RESULT_KINDS = frozenset({"result", "summary", "records"})

# Tags that mark system-managed memory notes; surfaced separately (not hidden —
# unlike the single-namespace ``memory_tools`` surface, the pool is a read-only
# audit/recall view, so seeing a skill body is fine) but flagged so callers can
# filter. Kept in sync with ``contractor.tools.memory._RESERVED_TAGS``.
_RESERVED_TAGS = frozenset({"skill", "inbox"})


@dataclass(frozen=True, slots=True)
class PoolKey:
    """A parsed artifact key: ``(namespace, kind)`` derived from the raw key."""

    raw: str
    namespace: str
    kind: PoolKind

    @classmethod
    def parse(cls, key: str) -> PoolKey:
        if key.startswith(_MEMORY_PREFIX):
            return cls(raw=key, namespace=key[len(_MEMORY_PREFIX):], kind="memory")
        head, sep, tail = key.rpartition("/")
        if sep and tail in _RESULT_KINDS:
            return cls(raw=key, namespace=head, kind=tail)  # type: ignore[arg-type]
        return cls(raw=key, namespace=key, kind="raw")

    def matches(self, mask: str) -> bool:
        """fnmatch the mask against the namespace, then the raw key.

        Namespace-first so developer masks read naturally
        (``trace_annotation:*`` hits ``user:memory/trace_annotation:...`` and
        ``trace_annotation/openapi/.../result`` alike); raw fallback lets a mask
        target bespoke keys (``oas-*``) or kinds (``*/records``).
        """
        return fnmatch.fnmatch(self.namespace, mask) or fnmatch.fnmatch(self.raw, mask)


@dataclass(frozen=True, slots=True)
class PoolDocument:
    """One searchable/indexable unit of the pool.

    A memory store expands into one document *per note* (so a namespace full of
    injected skill bodies doesn't swamp search as a single giant hit); every
    other artifact is one document carrying its whole body. ``doc_id`` is unique
    across the pool and stable, so an index can key on it.
    """

    key: str
    namespace: str
    kind: PoolKind
    text: str
    note_name: str | None = None

    @property
    def doc_id(self) -> str:
        return f"{self.key}#{self.note_name}" if self.note_name else self.key


@dataclass(frozen=True, slots=True)
class PoolHit:
    """One search result: a located entry plus a relevance score and snippet."""

    key: str
    namespace: str
    kind: PoolKind
    score: float
    snippet: str
    note_name: str | None = None


# ─── Pool reader (backend-agnostic) ───────────────────────────────────────────


@dataclass(slots=True)
class ArtifactPool:
    """Masked, read-only view over the artifact service.

    ``masks`` is an allowlist of fnmatch globs applied to every key (see
    :meth:`PoolKey.matches`); a key invisible under the masks cannot be listed,
    read, or searched. Default ``["*"]`` exposes the whole pool. Restrict it
    (e.g. ``["trace_annotation:*", "oas-*"]``) to fence a worker into the
    namespaces it should reach.
    """

    artifact_service: BaseArtifactService
    app_name: str
    user_id: str
    masks: tuple[str, ...] = ("*",)

    def _visible(self, pk: PoolKey) -> bool:
        return any(pk.matches(m) for m in self.masks)

    async def keys(self, mask: str = "*") -> list[PoolKey]:
        raw = await self.artifact_service.list_artifact_keys(
            app_name=self.app_name, user_id=self.user_id, session_id=None
        )
        return [
            pk
            for pk in (PoolKey.parse(k) for k in raw)
            if self._visible(pk) and pk.matches(mask)
        ]

    async def load_text(self, key: str) -> str | None:
        pk = PoolKey.parse(key)
        if not self._visible(pk):
            return None
        part = await self.artifact_service.load_artifact(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=None,
            filename=key,
        )
        if part is None:
            return None
        return part.text or ""

    async def load_notes(self, namespace: str) -> dict[str, dict[str, Any]]:
        """Parse a ``user:memory/<namespace>`` store into ``{name: note-dict}``."""
        text = await self.load_text(f"{_MEMORY_PREFIX}{namespace}")
        if not text:
            return {}
        raw = yaml.safe_load(text) or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    async def documents(
        self, mask: str = "*", *, include_reserved: bool = False
    ) -> list[PoolDocument]:
        """Expand the visible pool into searchable/indexable documents.

        Memory stores fan out to one document per note; ``skill``/``inbox``
        notes (injected reference bodies and cross-task plumbing — never
        run-specific knowledge) are dropped unless ``include_reserved``. Every
        other artifact is a single document.
        """
        docs: list[PoolDocument] = []
        for pk in await self.keys(mask):
            if pk.kind == "memory":
                notes = await self.load_notes(pk.namespace)
                for note_name, note in notes.items():
                    tags = note.get("tags") or []
                    if not include_reserved and _RESERVED_TAGS.intersection(tags):
                        continue
                    body = str(note.get("memory") or "")
                    if not body:
                        continue
                    docs.append(
                        PoolDocument(
                            key=pk.raw,
                            namespace=pk.namespace,
                            kind=pk.kind,
                            text=body,
                            note_name=note_name,
                        )
                    )
            else:
                body = await self.load_text(pk.raw)
                if body:
                    docs.append(
                        PoolDocument(
                            key=pk.raw, namespace=pk.namespace, kind=pk.kind, text=body
                        )
                    )
        return docs


# ─── Search backend (the RAG seam) ────────────────────────────────────────────


@runtime_checkable
class ArtifactPoolBackend(Protocol):
    """Pluggable search over the pool. Swap the impl, keep the tools.

    The default is :class:`KeywordPoolBackend`. A future embedding backend
    (in-memory numpy / FAISS) or ``PgVectorPoolBackend`` implements this same
    method — chunk + embed each entry on save, ANN-search here — and the
    frontend ``pool_search`` tool is unchanged.
    """

    async def search(
        self, pool: ArtifactPool, query: str, *, mask: str, k: int
    ) -> list[PoolHit]: ...


@dataclass(slots=True)
class KeywordPoolBackend:
    """Dependency-free term-frequency ranker with first-match snippets.

    Loads every visible entry's body and scores by summed case-insensitive
    occurrences of the query's whitespace-split terms. Fine for the hundreds of
    artifacts a run produces; replace with an embedding backend for recall at
    scale (see :class:`ArtifactPoolBackend`).
    """

    snippet_radius: int = 160

    async def search(
        self, pool: ArtifactPool, query: str, *, mask: str, k: int
    ) -> list[PoolHit]:
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        hits: list[PoolHit] = []
        for doc in await pool.documents(mask):
            low = doc.text.lower()
            score = sum(low.count(t) for t in terms)
            if score <= 0:
                continue
            pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
            start = max(0, pos - self.snippet_radius)
            snippet = doc.text[start : pos + self.snippet_radius].strip()
            hits.append(
                PoolHit(
                    key=doc.key,
                    namespace=doc.namespace,
                    kind=doc.kind,
                    score=float(score),
                    snippet=snippet,
                    note_name=doc.note_name,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


# ─── Frontend tool factory ────────────────────────────────────────────────────


def artifact_pool_tools(
    *,
    artifact_service: BaseArtifactService,
    app_name: str,
    user_id: str,
    masks: list[str] | None = None,
    backend: ArtifactPoolBackend | None = None,
) -> list:
    """Build the read-only ``pool_*`` tools bound to one artifact pool.

    Wire it into a ``build_<agent>`` factory exactly like ``memory_tools`` —
    pass ``artifact_service``/``app_name``/``user_id`` (available on
    ``WorkflowContext``) and an optional ``masks`` allowlist. ``backend``
    defaults to :class:`KeywordPoolBackend`; pass a RAG backend to upgrade
    search without changing callers.
    """
    pool = ArtifactPool(
        artifact_service=artifact_service,
        app_name=app_name,
        user_id=user_id,
        masks=tuple(masks or ("*",)),
    )
    search_backend = backend or KeywordPoolBackend()

    async def pool_namespaces(tool_context) -> dict[str, Any]:
        """Lists every namespace in the artifact pool with its entry kinds.

        Start here to discover what other tasks/workflows have produced.
        Namespaces group related artifacts (one per memory store, one per task
        output key). Use the returned namespace strings as the ``namespace``
        argument to pool_list / pool_read_memory, or as a mask prefix.

        Returns:
            A list of ``{namespace, kinds, count}`` entries.
        """

        async def _impl() -> Any:
            groups: dict[str, set[str]] = {}
            for pk in await pool.keys():
                groups.setdefault(pk.namespace, set()).add(pk.kind)
            rows = [
                {"namespace": ns, "kinds": sorted(kinds), "count": len(kinds)}
                for ns, kinds in sorted(groups.items())
            ]
            return ok(rows, total_items=len(rows))

        return await aguard(_impl)

    async def pool_list(mask: str, tool_context) -> dict[str, Any]:
        """Lists artifact keys across namespaces, filtered by an fnmatch mask.

        Args:
            mask: fnmatch glob applied to the namespace (then the raw key).
                Examples: ``"*"`` (everything), ``"trace_annotation:*"`` (one
                workflow's notes + results), ``"*/records"`` (all record
                artifacts), ``"oas-*"`` (bespoke OpenAPI artifacts).

        Returns:
            A (possibly truncated) page of ``{key, namespace, kind}`` entries;
            ``total_items`` is the true match count.
        """

        async def _impl() -> Any:
            rows = [
                {"key": pk.raw, "namespace": pk.namespace, "kind": pk.kind}
                for pk in await pool.keys(mask or "*")
            ]
            limit = get_settings().fs_max_read_lines or len(rows)
            return ok_page(rows[:limit], total=len(rows))

        return await aguard(_impl)

    async def pool_read(
        key: str, offset: int, limit: int, tool_context
    ) -> dict[str, Any]:
        """Reads an artifact body by its full key, with honest char-windowing.

        Use pool_list / pool_namespaces to find the key first. For memory
        namespaces prefer pool_read_memory (it parses individual notes).

        Args:
            key: The full artifact key, e.g.
                ``"trace_annotation/openapi/users/result"`` or
                ``"trace-openapi-diff"``.
            offset: Start character offset (0 for the beginning).
            limit: Max characters to return; ``0`` uses the configured cap.

        Returns:
            The requested window under ``result``; ``truncated`` is true when
            more remains beyond the window.
        """

        async def _impl() -> Any:
            body = await pool.load_text(key)
            if body is None:
                return err(f"artifact {key!r} not found or outside allowed masks")
            cap = limit if limit and limit > 0 else get_settings().fs_max_output
            start = max(0, offset)
            window = body[start : start + cap]
            return ok_page(window, total=len(body), returned=start + len(window))

        return await aguard(_impl)

    async def pool_read_memory(
        namespace: str, name: str, tool_context
    ) -> dict[str, Any]:
        """Reads memory notes from ANY namespace (not just the current task's).

        This is the cross-namespace complement to read_memory/list_memories,
        which only see the current task's namespace.

        Args:
            namespace: The memory namespace (from pool_namespaces).
            name: A specific note name; leave empty to list all notes in the
                namespace (previews).

        Returns:
            One note (with body) when ``name`` is given, else a preview list.
            ``reserved`` flags notes that are skills/inbox entries.
        """

        async def _impl() -> Any:
            notes = await pool.load_notes(namespace)
            if not notes:
                return err(f"memory namespace {namespace!r} is empty or not visible")
            if name:
                item = notes.get(name)
                if item is None:
                    return err(
                        f"note {name!r} not found in {namespace!r}",
                        available=sorted(notes)[:50],
                    )
                tags = item.get("tags") or []
                return ok(item, reserved=bool(_RESERVED_TAGS.intersection(tags)))
            previews = [
                {
                    "name": n,
                    "description": v.get("description", ""),
                    "tags": v.get("tags", []),
                    "reserved": bool(_RESERVED_TAGS.intersection(v.get("tags") or [])),
                }
                for n, v in notes.items()
            ]
            return ok_page(previews, total=len(previews))

        return await aguard(_impl)

    async def pool_search(
        query: str, mask: str, k: int, tool_context
    ) -> dict[str, Any]:
        """Searches artifact and memory bodies across namespaces for ``query``.

        Prefer this over scanning with pool_list/pool_read when looking for a
        concept ("BOLA on orders", "JWT validation") rather than a known key.

        Args:
            query: Free-text query (terms are matched independently).
            mask: fnmatch namespace mask to scope the search (``"*"`` = all).
            k: Max number of ranked hits to return (default 8 when ``0``).

        Returns:
            Ranked ``{key, namespace, kind, score, snippet}`` hits.
        """

        async def _impl() -> Any:
            hits = await search_backend.search(
                pool, query, mask=mask or "*", k=k if k and k > 0 else 8
            )
            return ok([_hit_to_dict(h) for h in hits], total_items=len(hits))

        return await aguard(_impl)

    return [pool_namespaces, pool_list, pool_read, pool_read_memory, pool_search]


def _hit_to_dict(h: PoolHit) -> dict[str, Any]:
    row = {
        "key": h.key,
        "namespace": h.namespace,
        "kind": h.kind,
        "score": h.score,
        "snippet": h.snippet,
    }
    if h.note_name:
        row["note_name"] = h.note_name
    return row
