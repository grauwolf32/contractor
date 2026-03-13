"""
GitlabFileSystem — fsspec-совместимая read-only файловая система для GitLab API v4.

- Единственная HTTP-зависимость: aiohttp (никакого requests)
- Все файлы загружаются асинхронно при инициализации и хранятся в памяти (bytes)
- Никаких tmpdir / файловой системы — безопасно для k8s readonly rootfs
- Полностью совместима с FsspecCoverageFileTools / file_tools()

"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import io
import logging
import posixpath
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from urllib.parse import quote as url_quote
from contractor.utils.fs import project_id_encoded

import aiohttp
import fsspec
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Pydantic Settings
# ------------------------------------------------------------------ #


class GitlabFileSystemSettings(BaseSettings):
    """
    Конфигурация GitlabFileSystem.

    Значения берутся из аргументов конструктора, затем из переменных окружения.

    Env-переменные (prefix ``GITLAB_FS_``):
        GITLAB_FS_GITLAB_URL, GITLAB_FS_PROJECT_ID, GITLAB_FS_REF,
        GITLAB_FS_PRIVATE_TOKEN, GITLAB_FS_OAUTH_TOKEN, GITLAB_FS_JOB_TOKEN,
        GITLAB_FS_PER_PAGE, GITLAB_FS_TIMEOUT, GITLAB_FS_MAX_CONCURRENT,
        GITLAB_FS_MAX_FILE_SIZE, GITLAB_FS_MAX_RETRIES,
        GITLAB_FS_RETRY_BACKOFF_FACTOR, GITLAB_FS_RETRY_STATUSES

    Также поддерживаются legacy env-переменные (без префикса):
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
        default="main",
        description="Branch, tag, or commit SHA.",
    )

    # -- auth (приоритет: private_token > oauth_token > job_token) --
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
        default=4,
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

    # -- ssl (не из env, только программно) --
    ssl: Optional[Any] = Field(default=None, exclude=True)

    # -- validators --
    @field_validator("gitlab_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def _resolve_legacy_env_tokens(self) -> "GitlabFileSystemSettings":
        """
        Если токены не заданы через prefixed env, пробуем legacy env
        (GITLAB_PRIVATE_TOKEN, GITLAB_OAUTH_TOKEN, CI_JOB_TOKEN).
        Pydantic validation_alias уже должен это покрыть,
        но здесь — fallback на случай если prefix перекрыл.
        """

        if not self.private_token:
            self.private_token = os.environ.get("GITLAB_PRIVATE_TOKEN")
        if not self.oauth_token:
            self.oauth_token = os.environ.get("GITLAB_OAUTH_TOKEN")
        if not self.job_token:
            self.job_token = os.environ.get("CI_JOB_TOKEN")
        return self

    @property
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

    @property
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
    Получает дерево репозитория, затем параллельно скачивает
    все файлы в память (bytes).  Каждый HTTP-запрос оборачивается
    в retry с exponential backoff.
    """

    def __init__(self, settings: GitlabFileSystemSettings, project_id: str) -> None:
        self.settings = settings
        self.project_id = project_id
        self.api_base = (
            f"{settings.gitlab_url}/api/v4"
            f"/projects/{project_id_encoded(self.project_id)}"
        )
        self.ref = settings.ref
        self.headers = settings.auth_headers
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
        read_body: bool = False,
    ) -> aiohttp.ClientResponse:
        """
        Выполнить HTTP-запрос с retry + exponential backoff.

        Для GET-запросов, получающих JSON, вызывающий код сам разбирает resp.
        Если ``read_body=True``, тело будет прочитано внутри context manager
        (нужно для raw-файлов).

        ВАЖНО: при ``read_body=False`` возвращается response, у которого
        тело ещё НЕ прочитано — вызывающий код ДОЛЖЕН вызвать await resp.json()
        или await resp.read() сам.

        Raises ``aiohttp.ClientResponseError`` если все попытки исчерпаны.
        """
        last_exc: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await session.request(method, url, params=params, ssl=self.ssl)

                if resp.status not in self._retry_statuses:
                    resp.raise_for_status()
                    return resp

                # retryable status
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

            # backoff
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

        # все попытки исчерпаны
        raise last_exc  # type: ignore[misc]

    # -------------------- tree -------------------- #

    async def _fetch_tree(
        self,
        session: aiohttp.ClientSession,
    ) -> List[Dict[str, Any]]:
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

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        file_path: str,
    ) -> Tuple[str, Optional[bytes], Optional[str]]:
        """Скачать один файл в память. Возвращает (path, data|None, error|None)."""
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

    # -------------------- orchestration -------------------- #

    async def load_all(self) -> Tuple[List[TreeEntry], List[str]]:
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            enable_cleanup_closed=True,
        )

        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout,
            connector=connector,
        ) as session:
            # 1. Дерево
            raw_tree = await self._fetch_tree(session)

            # 2. Разделить
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

            # 3. Параллельная загрузка
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._download_file(session, semaphore, fp) for fp in blob_paths]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Раскидать по записям
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

        return tree_entries, errors


def _run_async(coro_factory):
    """Запустить корутину из синхронного контекста (включая Jupyter)."""
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
    Конфигурация полностью определяется GitlabFileSystemSettings.
    """

    protocol = "gitlab"
    root_marker = ""

    def __init__(
        self,
        project_id: str,
        settings: Optional[GitlabFileSystemSettings] = None,
        *,
        autoload: bool = True,
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

        if autoload:
            self.reload()

    # ------------------------------------------------------------------ #
    #  Settings access
    # ------------------------------------------------------------------ #

    @property
    def settings(self) -> GitlabFileSystemSettings:
        return self._settings

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def _make_loader(self) -> GitlabAsyncLoader:
        return GitlabAsyncLoader(settings=self._settings, project_id=self.project_id)

    def reload(
        self,
        settings: Optional[GitlabFileSystemSettings] = None,
    ) -> None:
        """
        Полная перезагрузка файловой системы.

        Можно передать новый объект Settings.
        """

        if settings is not None:
            self._settings = settings

        self.gitlab_url = self._settings.gitlab_url
        self.ref = self._settings.ref

        loader = self._make_loader()
        tree_entries, self._errors = _run_async(loader.load_all)
        self._build_index(tree_entries)

    def _build_index(self, tree_entries: List[TreeEntry]) -> None:
        self._entries.clear()
        self._children.clear()

        # root
        root = TreeEntry(path="", name="", entry_type="tree")
        self._entries[""] = root
        self._children[""] = []

        for te in tree_entries:
            self._entries[te.path] = te

            parent = posixpath.dirname(te.path)
            self._children.setdefault(parent, []).append(te.path)

            if te.is_dir:
                self._children.setdefault(te.path, [])

    def _do_load(self) -> None:
        loader = self._make_loader()
        tree_entries, self._errors = _run_async(loader.load_all)
        self._build_index(tree_entries)

    def reload(self, **kwargs: Any) -> None:
        """
        Перезагрузить всё из GitLab.

        Можно передать любые параметры из Settings для переопределения:
            fs.reload(ref="develop", timeout=120)
        """
        if kwargs:
            current = self._settings.model_dump()
            current.update(kwargs)
            self._settings = GitlabFileSystemSettings(**current)
            self.gitlab_url = self._settings.gitlab_url
            self.ref = self._settings.ref

        self._do_load()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def load_errors(self) -> List[str]:
        return list(self._errors)

    @property
    def file_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_file)

    @property
    def dir_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_dir)

    @property
    def memory_usage(self) -> int:
        """Приблизительное потребление памяти (только содержимое файлов)."""
        return sum(
            len(e.data)
            for e in self._entries.values()
            if e.is_file and e.data is not None
        )

    # ------------------------------------------------------------------ #
    #  fsspec abstract interface
    # ------------------------------------------------------------------ #

    def _strip_protocol(self, path: str) -> str:  # type: ignore[override]
        if isinstance(path, list):
            return [self._strip_protocol(p) for p in path]  # type: ignore[return-value]
        return _normalize_path(str(path))

    def invalidate_cache(self, path: Optional[str] = None) -> None:
        pass  # in-memory — ничего инвалидировать не нужно

    # ---------- info ----------

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        norm = _normalize_path(path)
        entry = self._entries.get(norm)
        if entry is None:
            raise FileNotFoundError(f"Path not found: {path}")
        return {
            "name": norm or "/",
            "type": "directory" if entry.is_dir else "file",
            "size": entry.size if entry.is_file else 0,
        }

    # ---------- ls ----------

    def ls(self, path: str = "", detail: bool = False, **kwargs: Any) -> list:
        norm = _normalize_path(path)

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
                        "type": "directory" if child.is_dir else "file",
                        "size": child.size if child.is_file else 0,
                    }
                )
            else:
                results.append(child.path)

        return results

    # ---------- exists / isdir / isfile / size ----------

    def exists(self, path: str, **kwargs: Any) -> bool:
        return _normalize_path(path) in self._entries

    def isdir(self, path: str) -> bool:
        entry = self._entries.get(_normalize_path(path))
        return entry is not None and entry.is_dir

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
        norm = _normalize_path(path)

        entry = self._entries.get(norm)
        if entry is None or not entry.is_dir:
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
                if child.is_dir:
                    dir_names.append(child.name)
                    queue.append(child.path)
                else:
                    file_names.append(child.name)

            virtual = "/" + current if current else "/"
            yield virtual, dir_names, file_names

    # ---------- glob ----------

    def glob(self, path: str, **kwargs: Any) -> List[str]:
        pattern = _normalize_path(path)
        if not pattern:
            return []

        matches: List[str] = []
        for entry_path in self._entries:
            if not entry_path:
                continue
            if fnmatch.fnmatch(entry_path, pattern):
                matches.append("/" + entry_path)

        return sorted(set(matches))

    # ---------- file reading (in-memory) ----------

    def _get_file_entry(self, path: str) -> TreeEntry:
        norm = _normalize_path(path)
        entry = self._entries.get(norm)
        if entry is None:
            raise FileNotFoundError(f"File not found: {path}")
        if entry.is_dir:
            raise IsADirectoryError(f"Is a directory: {path}")
        if entry.data is None:
            raise FileNotFoundError(
                f"File content not available (skipped during load): {path}"
            )
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
        return (
            f"GitlabFileSystem("
            f"url={self.gitlab_url!r}, "
            f"project={project_id_encoded(self.project_id)}, "
            f"ref={self.ref!r}, "
            f"files={self.file_count}, "
            f"dirs={self.dir_count}, "
            f"mem={mb:.1f}MB, "
            f"retries={self._settings.max_retries})"
        )
