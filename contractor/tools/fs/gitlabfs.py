"""
GitlabFileSystem — fsspec-compatible read-only filesystem for GitLab API v4.

- Single HTTP dependency: aiohttp (no requests)
- Files are loaded asynchronously in a background worker thread
- While loading, grep/glob fall back to GitLab API
- No tmpdir / filesystem writes — safe for k8s readonly rootfs
- Fully compatible with FsspecCoverageFileTools / file_tools()
"""

from __future__ import annotations

import asyncio
import enum
import fnmatch
import os
import io
import logging
import posixpath
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from urllib.parse import quote as url_quote

import aiohttp
import fsspec
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Loading State
# ------------------------------------------------------------------ #


class LoadingState(enum.Enum):
    """State of the background file loader."""

    NOT_STARTED = "not_started"
    LOADING_TREE = "loading_tree"
    LOADING_FILES = "loading_files"
    READY = "ready"
    FAILED = "failed"


# ------------------------------------------------------------------ #
#  Pydantic Settings
# ------------------------------------------------------------------ #


class GitlabFileSystemSettings(BaseSettings):
    """
    Configuration for GitlabFileSystem.

    Values come from constructor args, then from environment variables.

    Env variables (prefix ``GITLAB_FS_``):
        GITLAB_FS_GITLAB_URL, GITLAB_FS_PROJECT_ID, GITLAB_FS_REF,
        GITLAB_FS_PRIVATE_TOKEN, GITLAB_FS_OAUTH_TOKEN, GITLAB_FS_JOB_TOKEN,
        GITLAB_FS_PER_PAGE, GITLAB_FS_TIMEOUT, GITLAB_FS_MAX_CONCURRENT,
        GITLAB_FS_MAX_FILE_SIZE, GITLAB_FS_MAX_RETRIES,
        GITLAB_FS_RETRY_BACKOFF_FACTOR, GITLAB_FS_RETRY_STATUSES

    Also supports legacy env variables (no prefix):
        GITLAB_PRIVATE_TOKEN, GITLAB_OAUTH_TOKEN, CI_JOB_TOKEN
    """

    model_config = SettingsConfigDict(
        env_prefix="GITLAB_FS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # -- connection --
    gitlab_url: str = Field(
        default="https://gitlab.com",
        description="Base URL of the GitLab instance.",
    )

    ref: str = Field(
        default="master",
        description="Branch, tag, or commit SHA.",
    )

    # -- auth (priority: private_token > oauth_token > job_token) --
    private_token: Optional[str] = Field(
        default=None,
        validation_alias="GITLAB_PRIVATE_TOKEN",
        description="GitLab personal/project access token.",
    )
    oauth_token: Optional[str] = Field(
        default=None,
        validation_alias="GITLAB_OAUTH_TOKEN",
        description="GitLab OAuth2 token.",
    )
    job_token: Optional[str] = Field(
        default=None,
        validation_alias="CI_JOB_TOKEN",
        description="CI/CD job token.",
    )

    # -- loader tuning --
    per_page: int = Field(default=100, ge=1, le=100)
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Total HTTP timeout in seconds.",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        description="Max parallel file downloads.",
    )
    max_file_size: int = Field(
        default=50 * 1024 * 1024,
        ge=0,
        description="Skip files larger than this (bytes).",
    )

    # -- retry --
    max_retries: int = Field(
        default=5,
        ge=0,
        description="Number of retry attempts for failed HTTP requests.",
    )
    retry_backoff_factor: float = Field(
        default=5,
        ge=0,
        description="Exponential backoff multiplier: sleep = factor * 2^attempt.",
    )
    retry_statuses: FrozenSet[int] = Field(
        default=frozenset({429, 500, 502, 503, 504}),
        description="HTTP status codes that trigger a retry.",
    )

    # -- ssl (not from env, programmatic only) --
    ssl: Optional[Any] = Field(default=None, exclude=True)

    # -- validators --
    @field_validator("gitlab_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def _resolve_legacy_env_tokens(self) -> "GitlabFileSystemSettings":
        if not self.private_token:
            self.private_token = os.environ.get("GITLAB_PRIVATE_TOKEN")
        if not self.oauth_token:
            self.oauth_token = os.environ.get("GITLAB_OAUTH_TOKEN")
        if not self.job_token:
            self.job_token = os.environ.get("CI_JOB_TOKEN")
        return self

    def auth_headers(self) -> Dict[str, str]:
        if self.private_token:
            return {"PRIVATE-TOKEN": self.private_token}
        if self.oauth_token:
            return {"Authorization": f"Bearer {self.oauth_token}"}
        if self.job_token:
            return {"JOB-TOKEN": self.job_token}
        return {}


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _project_id_encoded(project_id: str) -> str:
    """URL-encode project ID (handles 'group/project' -> 'group%2Fproject')."""
    return url_quote(str(project_id), safe="")


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    if path.startswith("gitlab://"):
        path = path[len("gitlab://") :]
    path = posixpath.normpath(path)
    if path == ".":
        path = ""
    return path.strip("/")


class TreeEntry:
    __slots__ = ("path", "name", "entry_type", "size", "data")

    def __init__(
        self,
        path: str,
        name: str,
        entry_type: str,
        size: int = 0,
        data: Optional[bytes] = None,
    ) -> None:
        self.path = path
        self.name = name
        self.entry_type = entry_type  # "blob" | "tree"
        self.size = size
        self.data = data

    def is_dir(self) -> bool:
        return self.entry_type == "tree"

    @property
    def is_file(self) -> bool:
        return self.entry_type == "blob"


# ------------------------------------------------------------------ #
#  Async loader with retry
# ------------------------------------------------------------------ #


class GitlabAsyncLoader:
    """
    Fetches the repository tree and then downloads all files in parallel
    into memory. Each HTTP request is wrapped with retry + exponential backoff.
    """

    def __init__(self, settings: GitlabFileSystemSettings, project_id: str) -> None:
        self.settings = settings
        self.project_id = project_id
        self.api_base = (
            f"{settings.gitlab_url}/api/v4"
            f"/projects/{_project_id_encoded(self.project_id)}"
        )
        self.ref = settings.ref
        self.headers = settings.auth_headers()
        self.per_page = settings.per_page
        self.timeout = aiohttp.ClientTimeout(total=settings.timeout)
        self.max_concurrent = settings.max_concurrent
        self.max_file_size = settings.max_file_size
        self.ssl = settings.ssl

        # retry
        self._max_retries = settings.max_retries
        self._backoff_factor = settings.retry_backoff_factor
        self._retry_statuses: Set[int] = set(settings.retry_statuses)

    # -------------------- retry wrapper -------------------- #

    async def _request_with_retry(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> aiohttp.ClientResponse:
        last_exc: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await session.request(method, url, params=params, ssl=self.ssl)

                if resp.status not in self._retry_statuses:
                    resp.raise_for_status()
                    return resp

                body_text = await resp.text()
                last_exc = aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=f"{resp.status}: {body_text[:200]}",
                )
                resp.release()

            except (
                aiohttp.ClientResponseError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last_exc = exc

            if attempt < self._max_retries:
                delay = self._backoff_factor * (2**attempt)
                logger.warning(
                    "Retry %d/%d for %s %s (reason: %s), sleeping %.2fs",
                    attempt + 1,
                    self._max_retries,
                    method,
                    url,
                    last_exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # -------------------- tree -------------------- #

    async def fetch_tree(
        self,
        session: aiohttp.ClientSession,
    ) -> List[Dict[str, Any]]:
        """Public: fetch full recursive tree. Used by both loader and fallback."""
        entries: List[Dict[str, Any]] = []
        page = 1

        while True:
            url = f"{self.api_base}/repository/tree"
            params: Dict[str, Any] = {
                "ref": self.ref,
                "recursive": "true",
                "per_page": self.per_page,
                "page": page,
            }

            resp = await self._request_with_retry(session, "GET", url, params=params)
            data = await resp.json()
            if not data:
                break

            entries.extend(data)

            total_pages = resp.headers.get("X-Total-Pages")
            if total_pages is not None and page >= int(total_pages):
                break
            if len(data) < self.per_page:
                break

            page += 1

        return entries

    # -------------------- single file -------------------- #

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        file_path: str,
    ) -> Tuple[str, Optional[bytes], Optional[str]]:
        """Download a single file into memory. Returns (path, data|None, error|None)."""
        encoded = url_quote(file_path, safe="")
        url = f"{self.api_base}/repository/files/{encoded}/raw"
        params = {"ref": self.ref}

        async with semaphore:
            try:
                resp = await self._request_with_retry(
                    session, "GET", url, params=params
                )

                cl = resp.headers.get("Content-Length")
                if cl and int(cl) > self.max_file_size:
                    resp.release()
                    return (file_path, None, f"too large: {cl} bytes")

                data = await resp.read()

                if len(data) > self.max_file_size:
                    return (file_path, None, f"too large: {len(data)} bytes")

                return (file_path, data, None)

            except aiohttp.ClientResponseError as exc:
                return (file_path, None, f"HTTP {exc.status}: {exc.message}")
            except asyncio.TimeoutError:
                return (file_path, None, "timeout (all retries exhausted)")
            except Exception as exc:  # noqa: BLE001
                return (file_path, None, f"{type(exc).__name__}: {exc}")

    # -------------------- single file (no semaphore) -------------------- #

    async def download_file_simple(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
    ) -> Tuple[str, Optional[bytes], Optional[str]]:
        """Download a single file — used by fallback, no semaphore needed."""
        encoded = url_quote(file_path, safe="")
        url = f"{self.api_base}/repository/files/{encoded}/raw"
        params = {"ref": self.ref}

        try:
            resp = await self._request_with_retry(session, "GET", url, params=params)
            data = await resp.read()
            return (file_path, data, None)
        except aiohttp.ClientResponseError as exc:
            return (file_path, None, f"HTTP {exc.status}: {exc.message}")
        except asyncio.TimeoutError:
            return (file_path, None, "timeout")
        except Exception as exc:  # noqa: BLE001
            return (file_path, None, f"{type(exc).__name__}: {exc}")

    # -------------------- search (grep via API) -------------------- #

    async def search_blobs(
        self,
        session: aiohttp.ClientSession,
        search_query: str,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Use GitLab search API to find blobs matching a query.
        GET /projects/:id/search?scope=blobs&search=<query>&ref=<ref>
        """
        results: List[Dict[str, Any]] = []
        page = 1

        while True:
            url = f"{self.api_base}/search"
            params: Dict[str, Any] = {
                "scope": "blobs",
                "search": search_query,
                "ref": self.ref,
                "per_page": per_page,
                "page": page,
            }

            try:
                resp = await self._request_with_retry(
                    session, "GET", url, params=params
                )
                data = await resp.json()
                if not data:
                    break

                results.extend(data)

                total_pages = resp.headers.get("X-Total-Pages")
                if total_pages is not None and page >= int(total_pages):
                    break
                if len(data) < per_page:
                    break

                page += 1
            except Exception as exc:
                logger.warning("GitLab search API error: %s", exc)
                break

        return results

    # -------------------- full load orchestration -------------------- #

    async def load_all(
        self,
        on_tree_ready: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_state_change: Optional[Callable[[LoadingState], None]] = None,
    ) -> Tuple[List[TreeEntry], List[str]]:
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            enable_cleanup_closed=True,
        )

        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout,
            connector=connector,
        ) as session:
            # 1. Tree
            if on_state_change:
                on_state_change(LoadingState.LOADING_TREE)

            raw_tree = await self.fetch_tree(session)

            # Notify tree is ready (so fallback can use it immediately)
            if on_tree_ready:
                on_tree_ready(raw_tree)

            # 2. Separate blobs and trees
            if on_state_change:
                on_state_change(LoadingState.LOADING_FILES)

            tree_entries: List[TreeEntry] = []
            blob_paths: List[str] = []

            for item in raw_tree:
                entry_path: str = item["path"]
                entry_type: str = item["type"]
                name = posixpath.basename(entry_path)

                te = TreeEntry(
                    path=entry_path,
                    name=name,
                    entry_type=entry_type,
                )
                tree_entries.append(te)

                if entry_type == "blob":
                    blob_paths.append(entry_path)

            # 3. Parallel download
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self.download_file(session, semaphore, fp) for fp in blob_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Map results
            data_map: Dict[str, bytes] = {}
            errors: List[str] = []

            for result in results:
                if isinstance(result, BaseException):
                    errors.append(str(result))
                    continue

                file_path, data, error = result
                if error:
                    errors.append(f"{file_path}: {error}")
                elif data is not None:
                    data_map[file_path] = data

            for te in tree_entries:
                if te.is_file and te.path in data_map:
                    te.data = data_map[te.path]
                    te.size = len(te.data)

            if on_state_change:
                on_state_change(LoadingState.READY)

            return tree_entries, errors


# ------------------------------------------------------------------ #
#  Background Worker
# ------------------------------------------------------------------ #


class _BackgroundLoader:
    """
    Manages the background loading thread and provides thread-safe
    access to loading state.
    """

    def __init__(self, fs: "GitlabFileSystem") -> None:
        self._fs = fs
        self._state = LoadingState.NOT_STARTED
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._tree_ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._errors: List[str] = []

        # Raw tree data available once tree is fetched (before files are loaded)
        self._raw_tree: Optional[List[Dict[str, Any]]] = None

    @property
    def state(self) -> LoadingState:
        with self._lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        return self.state == LoadingState.READY

    @property
    def is_loading(self) -> bool:
        return self.state in (LoadingState.LOADING_TREE, LoadingState.LOADING_FILES)

    @property
    def tree_available(self) -> bool:
        """True once the tree listing has been fetched (files may still be loading)."""
        return self._tree_ready_event.is_set()

    @property
    def raw_tree(self) -> Optional[List[Dict[str, Any]]]:
        return self._raw_tree

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until loading is complete. Returns True if ready."""
        return self._ready_event.wait(timeout=timeout)

    def wait_tree(self, timeout: Optional[float] = None) -> bool:
        """Block until tree listing is available."""
        return self._tree_ready_event.wait(timeout=timeout)

    def _on_tree_ready(self, raw_tree: List[Dict[str, Any]]) -> None:
        """Callback from loader when tree is fetched."""
        self._raw_tree = raw_tree

        # Build a partial index (tree structure only, no file data yet)
        self._fs._build_tree_index(raw_tree)
        self._tree_ready_event.set()
        logger.info(
            "Tree loaded: %d entries. File download in progress...",
            len(raw_tree),
        )

    def _on_state_change(self, new_state: LoadingState) -> None:
        with self._lock:
            self._state = new_state
        logger.debug("Loading state -> %s", new_state.value)

    def start(self) -> None:
        """Start background loading in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Background loader already running")
            return

        with self._lock:
            self._state = LoadingState.NOT_STARTED

        self._ready_event.clear()
        self._tree_ready_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name=f"gitlab-fs-loader-{self._fs.project_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Thread entry point."""
        try:
            loader = self._fs._make_loader()

            tree_entries, errors = _run_async(
                lambda: loader.load_all(
                    on_tree_ready=self._on_tree_ready,
                    on_state_change=self._on_state_change,
                )
            )

            self._errors = errors
            self._fs._build_index(tree_entries)

            with self._lock:
                self._state = LoadingState.READY

            self._tree_ready_event.set()  # ensure set even if tree callback missed
            self._ready_event.set()

            logger.info(
                "Background loading complete: %d files, %d errors",
                sum(1 for e in tree_entries if e.is_file),
                len(errors),
            )

        except Exception as exc:
            logger.error("Background loading failed: %s", exc, exc_info=True)
            with self._lock:
                self._state = LoadingState.FAILED
                self._errors = [str(exc)]
            self._tree_ready_event.set()  # unblock waiters
            self._ready_event.set()


# ------------------------------------------------------------------ #
#  GitLab API Fallback
# ------------------------------------------------------------------ #


class _GitlabApiFallback:
    """
    Provides grep and glob functionality via GitLab REST API
    when in-memory cache is not yet available.
    """

    def __init__(self, fs: "GitlabFileSystem") -> None:
        self._fs = fs

    def _make_session_and_loader(self) -> Tuple[GitlabAsyncLoader, Dict[str, Any]]:
        loader = self._fs._make_loader()
        return loader, {}

    # ---- glob via tree API ---- #

    def glob_via_api(self, pattern: str) -> List[str]:
        """
        Fetch the tree from GitLab API and apply fnmatch pattern.
        If tree is already available from background loader, use that.
        """
        bg = self._fs._bg_loader

        # If tree is available from background loader, just use it
        if bg is not None and bg.tree_available and bg.raw_tree is not None:
            return self._match_tree(bg.raw_tree, pattern)

        # Otherwise fetch tree via API
        raw_tree = self._fetch_tree_sync()
        return self._match_tree(raw_tree, pattern)

    def _match_tree(self, raw_tree: List[Dict[str, Any]], pattern: str) -> List[str]:
        norm_pattern = _normalize_path(pattern)
        matches: List[str] = []
        for item in raw_tree:
            entry_path = item["path"]
            if fnmatch.fnmatch(entry_path, norm_pattern):
                matches.append("/" + entry_path)
        return sorted(set(matches))

    def _fetch_tree_sync(self) -> List[Dict[str, Any]]:
        loader = self._fs._make_loader()

        async def _fetch():
            connector = aiohttp.TCPConnector(limit=5, enable_cleanup_closed=True)
            async with aiohttp.ClientSession(
                headers=loader.headers,
                timeout=loader.timeout,
                connector=connector,
            ) as session:
                return await loader.fetch_tree(session)

        return _run_async(_fetch)

    # ---- grep via search API ---- #

    def grep_via_api(
        self,
        pattern: str,
        path_pattern: Optional[str] = None,
        fixed_string: bool = False,
        max_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for pattern in files using GitLab search API.

        Returns list of dicts:
            [{"path": str, "line_number": int, "line": str, "match": str}, ...]
        """
        loader = self._fs._make_loader()

        async def _search():
            connector = aiohttp.TCPConnector(limit=5, enable_cleanup_closed=True)
            async with aiohttp.ClientSession(
                headers=loader.headers,
                timeout=loader.timeout,
                connector=connector,
            ) as session:
                raw_results = await loader.search_blobs(session, pattern)
                return self._process_search_results(
                    raw_results, pattern, path_pattern, fixed_string, max_count
                )

        return _run_async(_search)

    def _process_search_results(
        self,
        raw_results: List[Dict[str, Any]],
        pattern: str,
        path_pattern: Optional[str],
        fixed_string: bool,
        max_count: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Process GitLab search API results into a standardized format."""
        results: List[Dict[str, Any]] = []

        if fixed_string:

            def matcher(line):
                return pattern in line
        else:
            try:
                compiled = re.compile(pattern)

                def matcher(line):
                    return compiled.search(line) is not None
            except re.error:
                # If pattern is not valid regex, treat as fixed string
                def matcher(line):
                    return pattern in line

        for item in raw_results:
            file_path = item.get("filename", item.get("path", ""))

            # Filter by path pattern
            if path_pattern:
                norm_path_pattern = _normalize_path(path_pattern)
                if not fnmatch.fnmatch(file_path, norm_path_pattern):
                    continue

            # GitLab search returns 'data' with matched content
            data = item.get("data", "")
            startline = item.get("startline", 1)

            for i, line in enumerate(data.split("\n")):
                line_stripped = line.rstrip("\r")
                if matcher(line_stripped):
                    match_obj = {
                        "path": "/" + file_path,
                        "line_number": startline + i,
                        "line": line_stripped,
                    }

                    # Extract actual match
                    if not fixed_string:
                        try:
                            m = re.search(pattern, line_stripped)
                            match_obj["match"] = m.group(0) if m else line_stripped
                        except re.error:
                            match_obj["match"] = line_stripped
                    else:
                        match_obj["match"] = pattern

                    results.append(match_obj)

                    if max_count and len(results) >= max_count:
                        return results

        return results

    # ---- read single file via API ---- #

    def read_file_via_api(self, path: str) -> bytes:
        """Read a single file directly from GitLab API."""
        loader = self._fs._make_loader()

        async def _read():
            connector = aiohttp.TCPConnector(limit=5, enable_cleanup_closed=True)
            async with aiohttp.ClientSession(
                headers=loader.headers,
                timeout=loader.timeout,
                connector=connector,
            ) as session:
                _, data, error = await loader.download_file_simple(session, path)
                if error:
                    raise FileNotFoundError(f"Failed to read {path}: {error}")
                return data

        return _run_async(_read)


# ------------------------------------------------------------------ #
#  Async runner
# ------------------------------------------------------------------ #


def _run_async(coro_factory):
    """Run a coroutine from synchronous context (including Jupyter)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_factory())).result()
    else:
        return asyncio.run(coro_factory())


# ------------------------------------------------------------------ #
#  GitlabFileSystem
# ------------------------------------------------------------------ #


class GitlabFileSystem(fsspec.AbstractFileSystem):
    """
    Read-only in-memory fsspec filesystem backed by GitLab API v4.

    Features:
    - Background worker loads all files asynchronously
    - While loading, grep() and glob() fall back to GitLab API
    - File reads fall back to single-file API if cache not ready
    - Configuration is fully defined by GitlabFileSystemSettings

    Parameters
    ----------
    project_id : str
        GitLab project ID or "group/project" path.
    settings : GitlabFileSystemSettings, optional
        Configuration. If None, built from env vars.
    autoload : bool
        If True (default), start loading immediately.
    blocking : bool
        If True (default=False), block until all files are loaded.
        If False, load in background and use API fallback.
    """

    protocol = "gitlab"
    root_marker = ""

    def __init__(
        self,
        project_id: str,
        settings: Optional[GitlabFileSystemSettings] = None,
        *,
        autoload: bool = True,
        blocking: bool = False,
        **storage_options: Any,
    ) -> None:
        super().__init__(**storage_options)

        self._settings = settings or GitlabFileSystemSettings()
        self.project_id = project_id

        self.gitlab_url = self._settings.gitlab_url
        self.ref = self._settings.ref

        # state
        self._entries: Dict[str, TreeEntry] = {}
        self._children: Dict[str, List[str]] = {}
        self._errors: List[str] = []

        # tree-only paths (available before file content is loaded)
        self._tree_paths: Set[str] = set()

        # background loader
        self._bg_loader: Optional[_BackgroundLoader] = None
        self._fallback: _GitlabApiFallback = _GitlabApiFallback(self)

        if autoload:
            if blocking:
                # Synchronous full load (legacy behavior)
                self._do_load_sync()
            else:
                # Background load with API fallback
                self._start_background_load()

    # ------------------------------------------------------------------ #
    #  Settings access
    # ------------------------------------------------------------------ #

    @property
    def settings(self) -> GitlabFileSystemSettings:
        return self._settings

    @property
    def loading_state(self) -> LoadingState:
        """Current loading state."""
        if self._bg_loader is None:
            # Check if loaded synchronously
            if self._entries:
                return LoadingState.READY
            return LoadingState.NOT_STARTED
        return self._bg_loader.state

    @property
    def is_ready(self) -> bool:
        """True when all files are loaded into memory."""
        return self.loading_state == LoadingState.READY

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def _make_loader(self) -> GitlabAsyncLoader:
        return GitlabAsyncLoader(settings=self._settings, project_id=self.project_id)

    def _start_background_load(self) -> None:
        """Start loading files in a background thread."""
        self._bg_loader = _BackgroundLoader(self)
        self._bg_loader.start()

    def _do_load_sync(self) -> None:
        """Synchronous full load (blocking)."""
        loader = self._make_loader()
        tree_entries, self._errors = _run_async(loader.load_all)
        self._build_index(tree_entries)

    def reload(
        self,
        settings: Optional[GitlabFileSystemSettings] = None,
        blocking: bool = False,
    ) -> None:
        """
        Full filesystem reload.

        Parameters
        ----------
        settings : GitlabFileSystemSettings, optional
            New settings to use.
        blocking : bool
            If True, block until complete.
        """
        if settings is not None:
            self._settings = settings

        self.gitlab_url = self._settings.gitlab_url
        self.ref = self._settings.ref

        # Clear state
        self._entries.clear()
        self._children.clear()
        self._tree_paths.clear()
        self._errors.clear()

        if blocking:
            self._do_load_sync()
        else:
            self._start_background_load()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Block until background loading is complete.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait. None = wait forever.

        Returns
        -------
        bool
            True if loading completed, False if timed out.
        """
        if self._bg_loader is None:
            return self.is_ready
        return self._bg_loader.wait_ready(timeout=timeout)

    def _build_tree_index(self, raw_tree: List[Dict[str, Any]]) -> None:
        """
        Build a partial index from tree listing only (no file data).
        Called by background loader when tree is ready but files are still loading.
        Thread-safe: called from loader thread.
        """
        entries: Dict[str, TreeEntry] = {}
        children: Dict[str, List[str]] = {}
        tree_paths: Set[str] = set()

        # Root
        root = TreeEntry(path="", name="", entry_type="tree")
        entries[""] = root
        children[""] = []

        for item in raw_tree:
            entry_path = item["path"]
            entry_type = item["type"]
            name = posixpath.basename(entry_path)

            te = TreeEntry(path=entry_path, name=name, entry_type=entry_type)
            entries[entry_path] = te
            tree_paths.add(entry_path)

            parent = posixpath.dirname(entry_path)
            children.setdefault(parent, []).append(entry_path)

            if entry_type == "tree":
                children.setdefault(entry_path, [])

        # Atomic-ish swap (these are reference assignments, effectively atomic in CPython)
        self._entries = entries
        self._children = children
        self._tree_paths = tree_paths

    def _build_index(self, tree_entries: List[TreeEntry]) -> None:
        """Build full index with file data."""
        entries: Dict[str, TreeEntry] = {}
        children: Dict[str, List[str]] = {}

        # Root
        root = TreeEntry(path="", name="", entry_type="tree")
        entries[""] = root
        children[""] = []

        for te in tree_entries:
            entries[te.path] = te

            parent = posixpath.dirname(te.path)
            children.setdefault(parent, []).append(te.path)

            if te.is_dir():
                children.setdefault(te.path, [])

        # Atomic swap
        self._entries = entries
        self._children = children
        self._tree_paths = set(entries.keys())

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def load_errors(self) -> List[str]:
        if self._bg_loader and self._bg_loader._errors:
            return list(self._bg_loader._errors)
        return list(self._errors)

    @property
    def file_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_file)

    @property
    def dir_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_dir())

    @property
    def memory_usage(self) -> int:
        """Total memory consumption (file contents only)."""
        return sum(
            len(e.data)
            for e in self._entries.values()
            if e.is_file and e.data is not None
        )

    # ------------------------------------------------------------------ #
    #  Helper: get entry with fallback
    # ------------------------------------------------------------------ #

    def _has_file_data(self, norm_path: str) -> bool:
        """Check if we have file data in memory."""
        entry = self._entries.get(norm_path)
        return entry is not None and entry.is_file and entry.data is not None

    def _get_or_fetch_file_data(self, norm_path: str) -> bytes:
        """
        Get file data from memory cache, or fetch from API if not yet loaded.
        """
        entry = self._entries.get(norm_path)

        # If we have the data in memory, return it
        if entry is not None and entry.is_file and entry.data is not None:
            return entry.data

        # If loading and we know the path exists in tree
        if not self.is_ready and norm_path in self._tree_paths:
            logger.debug("Fallback: fetching %s via API (cache not ready)", norm_path)
            data = self._fallback.read_file_via_api(norm_path)
            return data

        # If loading hasn't fetched tree yet, try API directly
        if not self.is_ready and not self._tree_paths:
            logger.debug("Fallback: fetching %s via API (tree not ready)", norm_path)
            try:
                data = self._fallback.read_file_via_api(norm_path)
                return data
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise FileNotFoundError(
                    f"File not available and API fallback failed: {norm_path}: {exc}"
                )

        # Entry exists but no data (e.g., skipped during load)
        if entry is not None and entry.is_file and entry.data is None:
            if self.is_ready:
                raise FileNotFoundError(
                    f"File content not available (skipped during load): {norm_path}"
                )
            # Still loading — try API
            logger.debug(
                "Fallback: fetching %s via API (data not yet loaded)", norm_path
            )
            data = self._fallback.read_file_via_api(norm_path)
            return data

        raise FileNotFoundError(f"File not found: {norm_path}")

    # ------------------------------------------------------------------ #
    #  fsspec abstract interface
    # ------------------------------------------------------------------ #

    def _strip_protocol(self, path: str) -> str:  # type: ignore[override]
        if isinstance(path, list):
            return [self._strip_protocol(p) for p in path]  # type: ignore[return-value]
        return _normalize_path(str(path))

    def invalidate_cache(self, path: Optional[str] = None) -> None:
        pass  # in-memory — nothing to invalidate

    # ---------- info ----------

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        norm = _normalize_path(path)
        entry = self._entries.get(norm)
        if entry is None:
            # If still loading, check via API
            if not self.is_ready:
                # Wait for tree at least
                if self._bg_loader:
                    self._bg_loader.wait_tree(timeout=30)
                entry = self._entries.get(norm)

        if entry is None:
            raise FileNotFoundError(f"Path not found: {path}")

        return {
            "name": norm or "/",
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.size if entry.is_file else 0,
        }

    # ---------- ls ----------

    def ls(self, path: str = "", detail: bool = False, **kwargs: Any) -> list:
        norm = _normalize_path(path)

        # Wait for tree if loading
        if norm not in self._entries and not self.is_ready:
            if self._bg_loader:
                self._bg_loader.wait_tree(timeout=30)

        entry = self._entries.get(norm)
        if entry is None:
            raise FileNotFoundError(f"Path not found: {path}")

        if entry.is_file:
            info_dict = {
                "name": norm,
                "type": "file",
                "size": entry.size,
            }
            return [info_dict] if detail else [norm]

        children = self._children.get(norm, [])
        results: list = []

        for child_path in sorted(children):
            child = self._entries.get(child_path)
            if child is None:
                continue
            if detail:
                results.append(
                    {
                        "name": child.path,
                        "type": "directory" if child.is_dir() else "file",
                        "size": child.size if child.is_file else 0,
                    }
                )
            else:
                results.append(child.path)

        return results

    # ---------- exists / isdir / isfile / size ----------

    def exists(self, path: str, **kwargs: Any) -> bool:
        norm = _normalize_path(path)
        if norm in self._entries:
            return True
        # If still loading, wait for tree
        if not self.is_ready and self._bg_loader:
            self._bg_loader.wait_tree(timeout=30)
            return norm in self._entries
        return False

    def isdir(self, path: str) -> bool:
        entry = self._entries.get(_normalize_path(path))
        return entry is not None and entry.is_dir()

    def isfile(self, path: str) -> bool:
        entry = self._entries.get(_normalize_path(path))
        return entry is not None and entry.is_file

    def size(self, path: str) -> int:
        norm = _normalize_path(path)
        entry = self._entries.get(norm)
        if entry is None:
            raise FileNotFoundError(f"Path not found: {path}")
        return entry.size if entry.is_file else 0

    # ---------- walk ----------

    def walk(self, path: str = "", **kwargs: Any):
        # Ensure tree is loaded
        if not self.is_ready and self._bg_loader:
            self._bg_loader.wait_tree(timeout=60)

        norm = _normalize_path(path)
        entry = self._entries.get(norm)
        if entry is None or not entry.is_dir():
            return

        queue = [norm]
        visited: set = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            children = self._children.get(current, [])
            dir_names: List[str] = []
            file_names: List[str] = []

            for child_path in sorted(children):
                child = self._entries.get(child_path)
                if child is None:
                    continue
                if child.is_dir():
                    dir_names.append(child.name)
                    queue.append(child.path)
                else:
                    file_names.append(child.name)

            virtual = "/" + current if current else "/"
            yield virtual, dir_names, file_names

    # ---------- glob (with fallback) ----------

    def glob(self, path: str, **kwargs: Any) -> List[str]:
        pattern = _normalize_path(path)
        if not pattern:
            return []

        # If ready, use in-memory
        if self.is_ready:
            return self._glob_in_memory(pattern)

        # If tree is available, use it for matching
        if self._bg_loader and self._bg_loader.tree_available:
            return self._glob_in_memory(pattern)

        # Fallback to API
        logger.info("glob() falling back to GitLab API (cache not ready)")
        return self._fallback.glob_via_api(pattern)

    def _glob_in_memory(self, pattern: str) -> List[str]:
        matches: List[str] = []
        for entry_path in self._entries:
            if not entry_path:
                continue
            if fnmatch.fnmatch(entry_path, pattern):
                matches.append("/" + entry_path)
        return sorted(set(matches))

    # ---------- grep (with fallback) ----------

    def grep(
        self,
        pattern: str,
        path: str = "**/*",
        *,
        fixed_string: bool = False,
        max_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search file contents for a pattern.

        Falls back to GitLab Search API if in-memory cache is not ready.

        Parameters
        ----------
        pattern : str
            Regex pattern (or fixed string if fixed_string=True).
        path : str
            Glob pattern for file paths to search.
        fixed_string : bool
            If True, treat pattern as literal string.
        max_count : int, optional
            Maximum number of results.

        Returns
        -------
        list of dict
            Each dict has: path, line_number, line, match
        """
        if self.is_ready:
            return self._grep_in_memory(pattern, path, fixed_string, max_count)

        # Fallback to GitLab Search API
        logger.info("grep() falling back to GitLab Search API (cache not ready)")
        return self._fallback.grep_via_api(
            pattern=pattern,
            path_pattern=path if path != "**/*" else None,
            fixed_string=fixed_string,
            max_count=max_count,
        )

    def _grep_in_memory(
        self,
        pattern: str,
        path_glob: str,
        fixed_string: bool,
        max_count: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Search in-memory file contents."""
        norm_glob = _normalize_path(path_glob)
        results: List[Dict[str, Any]] = []

        if fixed_string:

            def matcher(line):
                return pattern in line
        else:
            try:
                compiled = re.compile(pattern)

                def matcher(line):
                    return compiled.search(line) is not None
            except re.error:

                def matcher(line):
                    return pattern in line

        for entry_path, entry in self._entries.items():
            if not entry_path or not entry.is_file or entry.data is None:
                continue

            if not fnmatch.fnmatch(entry_path, norm_glob):
                continue

            try:
                text = entry.data.decode("utf-8", errors="replace")
            except Exception:
                continue

            for line_no, line in enumerate(text.split("\n"), start=1):
                line_stripped = line.rstrip("\r")
                if matcher(line_stripped):
                    match_obj: Dict[str, Any] = {
                        "path": "/" + entry_path,
                        "line_number": line_no,
                        "line": line_stripped,
                    }

                    if not fixed_string:
                        try:
                            m = re.search(pattern, line_stripped)
                            match_obj["match"] = m.group(0) if m else line_stripped
                        except re.error:
                            match_obj["match"] = line_stripped
                    else:
                        match_obj["match"] = pattern

                    results.append(match_obj)

                    if max_count and len(results) >= max_count:
                        return results

        return results

    # ---------- file reading (in-memory with fallback) ----------

    def _get_file_entry(self, path: str) -> TreeEntry:
        """Get file entry — with API fallback for data."""
        norm = _normalize_path(path)
        entry = self._entries.get(norm)

        if entry is not None and entry.is_file and entry.data is not None:
            return entry

        # Try to fetch via API
        data = self._get_or_fetch_file_data(norm)

        # Create or update entry
        if entry is None:
            entry = TreeEntry(
                path=norm,
                name=posixpath.basename(norm),
                entry_type="blob",
                size=len(data),
                data=data,
            )
            self._entries[norm] = entry
        else:
            entry.data = data
            entry.size = len(data)

        return entry

    def _open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs: Any,
    ) -> io.IOBase:
        if "w" in mode or "a" in mode:
            raise NotImplementedError("GitlabFileSystem is read-only.")

        entry = self._get_file_entry(path)

        if "b" in mode:
            return io.BytesIO(entry.data)  # type: ignore[arg-type]
        else:
            encoding = kwargs.get("encoding", "utf-8")
            errors = kwargs.get("errors", "strict")
            text = entry.data.decode(encoding, errors=errors)  # type: ignore[union-attr]
            return io.StringIO(text)

    def cat_file(
        self,
        path: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> bytes:
        entry = self._get_file_entry(path)
        data = entry.data  # type: ignore[assignment]
        if start is not None or end is not None:
            data = data[start:end]
        return data  # type: ignore[return-value]

    def read_text(
        self,
        path: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        **kwargs: Any,
    ) -> str:
        return self.cat_file(path).decode(encoding, errors=errors)

    def head(self, path: str, size: int = 1024) -> bytes:
        return self.cat_file(path, start=0, end=size)

    def tail(self, path: str, size: int = 1024) -> bytes:
        data = self.cat_file(path)
        return data[-size:]

    # ------------------------------------------------------------------ #
    #  repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        mb = self.memory_usage / (1024 * 1024)
        state = self.loading_state.value
        return (
            f"GitlabFileSystem("
            f"url={self.gitlab_url!r}, "
            f"project={_project_id_encoded(self.project_id)}, "
            f"ref={self.ref!r}, "
            f"state={state}, "
            f"files={self.file_count}, "
            f"dirs={self.dir_count}, "
            f"mem={mb:.1f}MB, "
            f"retries={self._settings.max_retries})"
        )
