"""pgvector RAG backend for the artifact pool.

Implements :class:`~contractor.tools.artifact_pool.ArtifactPoolBackend` on top
of Postgres + the ``vector`` extension, so ``pool_search`` upgrades from the
default term-frequency ranker to dense semantic retrieval without any change to
the frontend tools or the agent.

Two halves:

* :meth:`PgVectorPoolBackend.index` — walk the pool, chunk each entry, embed via
  the LiteLLM proxy, and upsert into ``artifact_chunks`` (content-hash gated, so
  re-indexing an unchanged run is a no-op). Call it before search, or from a
  save-time hook.
* :meth:`PgVectorPoolBackend.search` — embed the query, ANN-search by cosine
  distance, post-filter by the pool's namespace masks, return ``PoolHit``s.

Dependencies are lazy: ``psycopg`` (v3) is imported only when the backend is
constructed, and embeddings go through ``litellm.aembedding`` (already a
dependency) — vectors are passed as ``%s::vector`` literals so the ``pgvector``
Python package is *not* required. Bring up the database with
``deploy/pgvector/deploy_pg.sh``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from contractor.tools.artifact_pool import ArtifactPool, PoolHit
from contractor.utils.settings import get_settings

if TYPE_CHECKING:
    from contractor.tools.artifact_pool import PoolDocument


def _to_vector_literal(vec: list[float]) -> str:
    """Render an embedding as a pgvector text literal (``[a,b,c]``)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _chunk(text: str, *, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping character windows."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


async def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Embed ``texts`` through the LiteLLM proxy.

    ``model`` defaults to ``Settings.rag_embedding_model``; the proxy base/key
    come from the same ``litellm_*`` settings the rest of the stack uses.
    """
    import litellm

    s = get_settings()
    resp = await litellm.aembedding(
        model=model or s.rag_embedding_model,
        input=texts,
        api_base=s.litellm_api_base,
        api_key=s.litellm_api_key,
    )
    # litellm normalizes to the OpenAI shape: data[i]["embedding"].
    return [row["embedding"] for row in resp["data"]]


@dataclass(slots=True)
class PgVectorPoolBackend:
    """Semantic search backend over an artifact pool, backed by pgvector.

    Construct with an explicit ``dsn`` or let it fall back to
    ``Settings.rag_db_dsn``. ``dim`` must match the embedding model's output
    dimension and the ``vector(dim)`` column created in the schema.
    """

    dsn: str | None = None
    embedding_model: str | None = None
    dim: int = 1024
    table: str = "artifact_chunks"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    _ensured: bool = field(default=False, init=False)

    def _resolved_dsn(self) -> str:
        dsn = self.dsn or get_settings().rag_db_dsn
        if not dsn:
            raise RuntimeError(
                "no pgvector DSN configured — set RAG_DB_DSN in cli/.env or pass "
                "dsn=... (see deploy/pgvector/deploy_pg.sh)"
            )
        return dsn

    async def _connect(self):  # noqa: ANN202 - psycopg type is lazy-imported
        try:
            import psycopg
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "psycopg is required for the pgvector backend: "
                "`poetry install --extras rag` (or `pip install 'psycopg[binary]'`)"
            ) from exc
        return await psycopg.AsyncConnection.connect(self._resolved_dsn())

    async def ensure_schema(self) -> None:
        """Create the ``vector`` extension, table, and ANN index if absent."""
        if self._ensured:
            return
        async with await self._connect() as conn, conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id           BIGSERIAL PRIMARY KEY,
                    app_name     TEXT NOT NULL,
                    user_id      TEXT NOT NULL,
                    doc_id       TEXT NOT NULL,
                    key          TEXT NOT NULL,
                    namespace    TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    note_name    TEXT,
                    chunk_idx    INT  NOT NULL,
                    content_hash TEXT NOT NULL,
                    body         TEXT NOT NULL,
                    embedding    vector({self.dim}) NOT NULL,
                    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (app_name, user_id, doc_id, chunk_idx)
                )
                """
            )
            await cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.commit()
        self._ensured = True

    @staticmethod
    def _hash(body: str) -> str:
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    async def index(self, pool: ArtifactPool, *, mask: str = "*") -> dict[str, int]:
        """Embed and upsert every pool *document*. Returns simple counters.

        Indexes the same documents search ranks — one per memory note (minus
        injected skill/inbox notes), one per other artifact — so embeddings stay
        at note granularity. Content-hash gated per document: an unchanged
        document is skipped, so re-indexing a stable pool only pays the hash
        comparison, not re-embedding.
        """
        await self.ensure_schema()
        indexed = skipped = 0
        async with await self._connect() as conn:
            for doc in await pool.documents(mask):
                digest = self._hash(doc.text)
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SELECT content_hash FROM {self.table} "
                        f"WHERE app_name=%s AND user_id=%s AND doc_id=%s LIMIT 1",
                        (pool.app_name, pool.user_id, doc.doc_id),
                    )
                    row = await cur.fetchone()
                    if row and row[0] == digest:
                        skipped += 1
                        continue
                    await self._upsert_document(conn, pool, doc, digest)
                    indexed += 1
            await conn.commit()
        return {"indexed": indexed, "skipped": skipped}

    async def _upsert_document(
        self, conn: Any, pool: ArtifactPool, doc: PoolDocument, digest: str
    ) -> None:
        chunks = _chunk(doc.text, size=self.chunk_size, overlap=self.chunk_overlap)
        embeddings = await embed_texts(chunks, model=self.embedding_model)
        async with conn.cursor() as cur:
            await cur.execute(
                f"DELETE FROM {self.table} "
                f"WHERE app_name=%s AND user_id=%s AND doc_id=%s",
                (pool.app_name, pool.user_id, doc.doc_id),
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                await cur.execute(
                    f"""
                    INSERT INTO {self.table}
                        (app_name, user_id, doc_id, key, namespace, kind,
                         note_name, chunk_idx, content_hash, body, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        pool.app_name,
                        pool.user_id,
                        doc.doc_id,
                        doc.key,
                        doc.namespace,
                        doc.kind,
                        doc.note_name,
                        idx,
                        digest,
                        chunk,
                        _to_vector_literal(emb),
                    ),
                )

    async def search(
        self, pool: ArtifactPool, query: str, *, mask: str, k: int
    ) -> list[PoolHit]:
        await self.ensure_schema()
        (query_emb,) = await embed_texts([query], model=self.embedding_model)
        # Over-fetch, then apply the pool's fnmatch masks + the per-call mask in
        # Python (fnmatch is not SQL-expressible) and dedupe to one hit per key.
        from contractor.tools.artifact_pool import PoolKey

        async with await self._connect() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT doc_id, key, namespace, kind, note_name, body,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table}
                WHERE app_name=%s AND user_id=%s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    _to_vector_literal(query_emb),
                    pool.app_name,
                    pool.user_id,
                    _to_vector_literal(query_emb),
                    max(k * 4, k),
                ),
            )
            rows = await cur.fetchall()

        seen: set[str] = set()
        hits: list[PoolHit] = []
        for doc_id, key, namespace, kind, note_name, body, score in rows:
            pk = PoolKey.parse(key)
            if doc_id in seen or not pool._visible(pk) or not pk.matches(mask):
                continue
            seen.add(doc_id)
            hits.append(
                PoolHit(
                    key=key,
                    namespace=namespace,
                    kind=kind,
                    score=float(score),
                    snippet=(body or "")[:320].strip(),
                    note_name=note_name,
                )
            )
            if len(hits) >= k:
                break
        return hits
