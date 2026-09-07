"""Construction and rollback of allocation-local ADK workers."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from google.adk.models.base_llm import BaseLlm

from contractor_runtime.agent_skills.runtime import (
    prepare_agent_skills,
    probe_native_agent_skills,
)
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.llm.factory import ModelFactory, gateway_model
from contractor_runtime.llm.openai import (
    OpenAICompatibleGatewayLlm,
)
from contractor_runtime.projectfs import (
    OverlayWorkspaceSession,
    WorkspaceAutoExporter,
)
from contractor_runtime.worker.runtime import AdkWorkerRuntime

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext


class AdkWorkerRuntimeFactory:
    ref = "adk@1"
    supports_agent_skills = True
    supports_worker_completion = True

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        artifact_client_factory: Callable[[str, Any], ArtifactClient] | None = None,
    ) -> None:
        self._model_factory = model_factory or gateway_model
        self._artifact_client_factory = artifact_client_factory

    async def probe(self) -> bool:
        # Gateway/model availability remains allocation-scoped, but advertising
        # adk@1 also promises the exact native Agent Skills APIs used below.
        return await probe_native_agent_skills()

    async def create(self, context: WorkerBuildContext) -> AdkWorkerRuntime:
        prepared = await prepare_agent_skills(
            context.resolved_skills,
            allocation_id=context.allocation_id,
            runtime_settings=context.runtime_settings,
            workspace=context.workspace,
            artifact_client_factory=self._artifact_client_factory,
        )
        if prepared is not None:
            context = replace(context, agent_skills=prepared)
        runtime: AdkWorkerRuntime | None = None
        model: BaseLlm | None = None
        try:
            workspace_exporter: WorkspaceAutoExporter | None = None
            if context.workspace_export is not None:
                if (
                    not isinstance(context.project_workspace, OverlayWorkspaceSession)
                    or self._artifact_client_factory is None
                ):
                    raise RuntimeError("overlay workspace export dependencies are unavailable")
                workspace_exporter = WorkspaceAutoExporter(
                    workspace=context.project_workspace,
                    client=self._artifact_client_factory(
                        context.allocation_id, context.runtime_settings
                    ),
                    namespace=context.namespace,
                    slots=context.workspace_export,
                )
            model = self._model_factory(context)
            runtime = AdkWorkerRuntime(
                context,
                model,
                model_factory=self._model_factory,
                workspace_exporter=workspace_exporter,
            )
            await runtime.start()
            return runtime
        except asyncio.CancelledError:
            if runtime is not None:
                with contextlib.suppress(Exception):
                    await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))
            else:
                if isinstance(model, OpenAICompatibleGatewayLlm):
                    with contextlib.suppress(Exception):
                        await model.close()
                if prepared is not None:
                    await prepared.close()
            raise
        except Exception:
            if runtime is not None:
                with contextlib.suppress(Exception):
                    await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))
            else:
                if isinstance(model, OpenAICompatibleGatewayLlm):
                    with contextlib.suppress(Exception):
                        await model.close()
                if prepared is not None:
                    await prepared.close()
            raise
