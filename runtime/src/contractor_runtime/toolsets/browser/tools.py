"""Model-facing browser controls; artifacts use the ordinary exact-write path."""

from __future__ import annotations

import asyncio
from types import MappingProxyType

from google.adk.tools.tool_context import ToolContext
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_metrics import RecordedToolCall
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.toolsets.browser.session import (
    BROWSER_ARGS,
    MAX_MAP_BYTES,
    MAX_SCREENSHOT_BYTES,
    MAX_SCREENSHOTS,
    BrowserSession,
)
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.artifact_write import (
    definite_write_rejection,
    write_artifact_exact,
)
from contractor_runtime.toolsets.common.artifacts import _reject_unconfigured_client
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.tool_base import SessionArtifactTool
from contractor_runtime.toolsets.common.versions import ModelArtifacts


class BrowserToolsetFactory:
    ref = "browser-tools@1"
    exported_tools = frozenset(
        {
            "browser_open",
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_fill",
            "browser_requests",
            "browser_screenshot",
            "browser_export_map",
        }
    )
    infrastructure_channels = MappingProxyType(
        {name: frozenset({"runtime-http-client"}) for name in exported_tools}
    )

    def __init__(self, client_factory=None):
        self._client_factory = client_factory or _reject_unconfigured_client

    async def probe(self):
        try:
            async with asyncio.timeout(20):
                async with async_playwright() as pw:
                    browser = await pw.chromium.launch(
                        headless=True, args=BROWSER_ARGS, proxy={"server": "http://127.0.0.1:9"}
                    )
                    await browser.close()
        except (PlaywrightError, OSError, TimeoutError):
            return frozenset()
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected,
        allocation_id,
        run_id,
        namespace,
        runtime_settings,
        workspace,
        resources,
        adapter_handles=None,
        project_workspace=None,
        controlled_http=None,
    ):
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(resources, self.ref)
        client = ModelArtifacts(
            self._client_factory(allocation_id, runtime_settings), resources.selected_versions
        )
        session = BrowserSession(
            controlled_http,
            resources.http_captures,
            timeout=runtime_settings.request_timeout_seconds,
        )
        constructors = {
            cls.name: cls
            for cls in (
                BrowserOpen,
                BrowserNavigate,
                BrowserSnapshot,
                BrowserClick,
                BrowserFill,
                BrowserRequests,
                BrowserScreenshot,
                BrowserExportMap,
            )
        }
        return {
            name: constructors[name](
                session,
                client,
                metrics,
                runtime_secrets(runtime_settings.model_copy(update={"http_identities": None})),
                namespace,
            )
            for name in selected
        }


class BrowserTool(SessionArtifactTool):
    def __init__(self, session, client, metrics, secrets, namespace):
        super().__init__(session, client, metrics, secrets)
        self.namespace = namespace

    def _recorded(self, arguments):
        return RecordedToolCall(
            self._metrics,
            self.name,
            arguments,
            secrets=(*self._secrets, *self._session.transport.private_values()),
        )

    async def execute(self, arguments, operation):
        added = self._session.transport.own_invocation()
        try:
            async with asyncio.timeout(self._session.timeout + 5):
                return await self._call(arguments, operation(), lambda result: result)
        except (PlaywrightError, TimeoutError):
            # Playwright exceptions contain source, selectors and sometimes
            # browser launch arguments. Keep private capabilities out of errors.
            raise ToolInputError(
                "Browser operation incomplete; inspect observations before retrying"
            ) from None
        finally:
            self._session.transport.finish_invocation(added)

    async def save(self, name, data, media_type, maximum):
        require_model_visible_binding(self.namespace, name)
        target = ArtifactRef(namespace=self.namespace, name=name)
        expected = self._client.versions.base(self.namespace, name)
        self._client.versions.admit_write(target)
        result = await write_artifact_exact(
            self._client.client,
            target,
            data=data,
            media_type=media_type,
            expected_revision=expected,
            max_bytes=maximum,
            reject_first=definite_write_rejection,
            reject_replay=lambda error: (
                definite_write_rejection(error) and error.code != "artifact_conflict"
            ),
        )
        self._client.versions.observe(result.artifact)
        return {
            "artifact": result.artifact.model_dump(by_alias=True),
            "mediaType": result.media_type,
            "size": len(data),
        }


class BrowserOpen(BrowserTool):
    name = "browser_open"
    description = (
        "Open an isolated browser context. Optional credential_id must be assigned by "
        "engagement. Returns context_id, safe credential catalog and allowed scope. Up to "
        "four contexts; no target traffic until navigation. "
    )

    async def __call__(self, credential_id: str | None = None) -> dict:
        return await self.execute(
            {"credential_id": credential_id}, lambda: self._session.open(credential_id)
        )


class BrowserNavigate(BrowserTool):
    name = "browser_navigate"
    description = (
        "Navigate context_id to an authorized HTTP(S) URL. Records a bounded action window "
        "and requests. Every resource and redirect passes engagement policy. An incomplete "
        "action may have executed; inspect before repeating. "
    )

    async def __call__(
        self, context_id: str, url: str, tool_context: ToolContext | None = None
    ) -> dict:
        return await self.execute(
            {"context_id": context_id, "url": url},
            lambda: self._session.action(
                context_id, "navigate", url, tool_context.invocation_id if tool_context else None
            ),
        )


class BrowserClick(BrowserTool):
    name = "browser_click"
    description = (
        "Click one element identified by a Playwright locator selector, with a bounded "
        "temporal request observation window. Mutating UI actions may execute. Never "
        "automatically retry incomplete actions. "
    )

    async def __call__(
        self, context_id: str, selector: str, tool_context: ToolContext | None = None
    ) -> dict:
        _selector(selector)
        return await self.execute(
            {"context_id": context_id, "selector": selector},
            lambda: self._session.action(
                context_id, "click", selector, tool_context.invocation_id if tool_context else None
            ),
        )


class BrowserFill(BrowserTool):
    name = "browser_fill"
    description = (
        "Fill one input. Value is excluded from browser action maps and tool metrics. Input "
        "events can trigger requests; inspect before repeating an incomplete action. "
    )

    async def __call__(
        self, context_id: str, selector: str, value: str, tool_context: ToolContext | None = None
    ) -> dict:
        _selector(selector)
        if not isinstance(value, str) or len(value) > 4096:
            raise ToolInputError("Input value must be a string of at most 4096 characters")
        return await self.execute(
            {"context_id": context_id, "selector": selector},
            lambda: self._session.action(
                context_id,
                "fill",
                (selector, value),
                tool_context.invocation_id if tool_context else None,
            ),
        )


class BrowserSnapshot(BrowserTool):
    name = "browser_snapshot"
    description = (
        "Read bounded visible text, links and control descriptions (100 links/controls, 12000 "
        "text characters). Input values are omitted. Use selectors from this observation for "
        "UI actions. "
    )

    async def __call__(self, context_id: str) -> dict:
        return await self.execute(
            {"context_id": context_id}, lambda: self._session.snapshot(context_id)
        )


class BrowserRequests(BrowserTool):
    name = "browser_requests"
    description = (
        "Read up to 100 browser request observations from offset. request_id selects captured "
        "HTTP evidence in the ordinary evidence tool. action_id means observed during its "
        "window; background requests have no action. Bodies and private credential headers "
        "are excluded. "
    )

    async def __call__(self, offset: int = 0) -> dict:
        if type(offset) is not int or offset < 0 or offset > 1024:
            raise ToolInputError("Offset must be an integer between zero and 1024")

        async def read():
            rows = self._session.requests
            return {
                "requests": rows[offset : offset + 100],
                "total": len(rows),
                "gaps": self._session.gaps,
            }

        return await self.execute({"offset": offset}, read)


class BrowserScreenshot(BrowserTool):
    name = "browser_screenshot"
    description = (
        "Save the current 1280x720 viewport as a PNG artifact (max 4 MiB, 64 captures). "
        "Password inputs are masked. Returns an exact ordinary artifact usable by "
        "finding(sources=[{kind: artifact, namespace: ..., name: ...}]). The image may "
        "contain target data; capture only relevant views. "
    )

    async def __call__(self, context_id: str, name: str) -> dict:
        async def capture():
            if self._session.screenshot_attempts >= MAX_SCREENSHOTS:
                raise ToolInputError("Screenshot capture limit reached")
            # Reserve before the first await: parallel calls cannot over-admit.
            # An ambiguous/failed publication never authorizes another capture.
            self._session.screenshot_attempts += 1
            data, url = await self._session.screenshot(context_id)
            result = await self.save(name, data, "image/png", MAX_SCREENSHOT_BYTES)
            last = next(
                (a["id"] for a in reversed(self._session.actions) if a["context_id"] == context_id),
                None,
            )
            self._session.screenshots.append(
                {"context_id": context_id, "url": url, "after_action_id": last, **result}
            )
            return result

        return await self.execute({"context_id": context_id, "name": name}, capture)


class BrowserExportMap(BrowserTool):
    name = "browser_export_map"
    description = (
        "Save Runtime-assembled frontend route/action/request/screenshot observations and "
        'explicit gaps as JSON (max 1 MiB). Call after exploration with name="route-map". '
        "No model-authored map or causal claims are accepted. "
    )

    async def __call__(self, name: str = "route-map") -> dict:
        async def export():
            return await self.save(name, self._session.export(), "application/json", MAX_MAP_BYTES)

        return await self.execute({"name": name}, export)


def _selector(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ToolInputError("Selector must contain between 1 and 512 characters")
