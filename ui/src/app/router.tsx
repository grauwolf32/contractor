import { createBrowserRouter, type RouteObject } from "react-router";

import { AuthenticatedRoute } from "../routes/guard";
import { HomeRoute } from "../routes/home";
import { LoginRoute } from "../routes/login";
import { ArtifactDetailRoute } from "../routes/artifacts/detail";
import { ArtifactListRoute } from "../routes/artifacts/list";
import {
  EvaluationArtifactDetailRoute,
  ProjectArtifactDetailRoute,
} from "../routes/projects/artifact-detail";
import {
  EvaluationDetailRoute,
  ProjectDetailRoute,
} from "../routes/projects/detail";
import { EvaluationListRoute, ProjectListRoute } from "../routes/projects/list";
import { ProjectAuditDetailRoute } from "../routes/projects/audits/detail";
import { ProjectAuditListRoute } from "../routes/projects/audits/list";
import { ProjectWorkflowRunRoute } from "../routes/projects/workflow-run";
import { RunArtifactDetailRoute } from "../routes/runs/artifacts";
import { RunDetailRoute } from "../routes/runs/detail";
import { LegacyQueueRedirect, RunsRoute } from "../routes/runs";
import {
  LegacyRuntimeConfigurationRedirect,
  RunConfigurationLayout,
} from "../routes/runs/configuration";
import { SkillsRoute } from "../routes/skills";
import {
  CatalogIndexRedirect,
  CatalogLayoutRoute,
  LegacyCatalogRedirect,
} from "../routes/catalog/layout";
import { AgentListRoute } from "../routes/catalog/agents";
import { AgentDetailRoute } from "../routes/catalog/agent-detail";
import { AllocationListRoute } from "../routes/operations/allocations";
import { CompletedAllocationListRoute } from "../routes/operations/allocations/completed";
import { CredentialDetailRoute } from "../routes/operations/credentials/detail";
import { CredentialListRoute } from "../routes/operations/credentials";
import { OperationsLayoutRoute } from "../routes/operations/layout";
import { OperationsOverviewRoute } from "../routes/operations/overview";
import { OperationsPerformanceRoute } from "../routes/operations/performance";
import { ConfigurationDetailRoute } from "../routes/operations/llm-configurations/detail";
import { ConfigurationListRoute } from "../routes/operations/llm-configurations";
import { RuntimeAgentListRoute } from "../routes/operations/runtime-agents";
import { RuntimeConfigDetailRoute } from "../routes/operations/runtime-configs/detail";
import { RuntimeConfigurationRoute } from "../routes/operations/runtime-configs";
import { OperationsSettingsRoute } from "../routes/operations/settings";
import { WorkflowDetailRoute } from "../routes/workflows/detail";
import { WorkflowListRoute } from "../routes/workflows/list";
import { NotFoundRoute } from "../routes/placeholders";
import { ApplicationShell } from "./shell";

export function applicationRoutes(): RouteObject[] {
  return [
    { path: "/login", element: <LoginRoute /> },
    {
      element: <AuthenticatedRoute />,
      children: [
        {
          element: <ApplicationShell />,
          children: [
            { index: true, element: <HomeRoute /> },
            {
              path: "/workflows",
              element: <LegacyCatalogRedirect />,
            },
            {
              path: "/workflows/:name/:version",
              element: <LegacyCatalogRedirect />,
            },
            {
              path: "/projects",
              element: <ProjectListRoute />,
            },
            {
              path: "/projects/:projectId",
              element: <ProjectDetailRoute />,
            },
            {
              path: "/projects/:projectId/artifacts/:namespace/:name",
              element: <ProjectArtifactDetailRoute />,
            },
            {
              path: "/projects/:projectId/workflows/:name/:version/run",
              element: <ProjectWorkflowRunRoute />,
            },
            {
              path: "/projects/:projectId/audits",
              element: <ProjectAuditListRoute />,
            },
            {
              path: "/projects/:projectId/audits/:auditId",
              element: <ProjectAuditDetailRoute />,
            },
            {
              path: "/projects/:projectId/audits/:auditId/:section",
              element: <ProjectAuditDetailRoute />,
            },
            { path: "/evals", element: <EvaluationListRoute /> },
            {
              path: "/evals/:projectId",
              element: <EvaluationDetailRoute />,
            },
            {
              path: "/evals/:projectId/artifacts/:namespace/:name",
              element: <EvaluationArtifactDetailRoute />,
            },
            { path: "/queue", element: <LegacyQueueRedirect /> },
            {
              path: "/artifacts",
              element: <ArtifactListRoute />,
            },
            {
              path: "/artifacts/:namespace/:name",
              element: <ArtifactDetailRoute />,
            },
            {
              path: "/runs",
              element: <RunsRoute />,
              children: [
                {
                  path: "configuration",
                  element: <RunConfigurationLayout />,
                  children: [
                    { index: true, element: <RuntimeConfigurationRoute /> },
                    {
                      path: ":name/:version",
                      element: <RuntimeConfigDetailRoute />,
                    },
                  ],
                },
              ],
            },
            {
              path: "/operations/runtime-configs",
              element: <LegacyRuntimeConfigurationRedirect />,
            },
            {
              path: "/operations/runtime-configs/:name/:version",
              element: <LegacyRuntimeConfigurationRedirect />,
            },
            { path: "/runs/:runId", element: <RunDetailRoute /> },
            {
              path: "/runs/:runId/artifacts/:namespace/:name",
              element: <RunArtifactDetailRoute />,
            },
            { path: "/skills", element: <LegacyCatalogRedirect /> },
            {
              path: "/catalog",
              element: <CatalogLayoutRoute />,
              children: [
                { index: true, element: <CatalogIndexRedirect /> },
                { path: "workflows", element: <WorkflowListRoute /> },
                {
                  path: "workflows/:name/:version",
                  element: <WorkflowDetailRoute />,
                },
                { path: "agents", element: <AgentListRoute /> },
                {
                  path: "agents/:name/:version",
                  element: <AgentDetailRoute />,
                },
                { path: "skills", element: <SkillsRoute /> },
              ],
            },
            {
              path: "/operations",
              element: <OperationsLayoutRoute />,
              children: [
                { index: true, element: <OperationsOverviewRoute /> },
                {
                  path: "runtime-agents",
                  element: <RuntimeAgentListRoute />,
                },
                {
                  path: "allocations",
                  element: <AllocationListRoute />,
                },
                {
                  path: "allocations/completed",
                  element: <CompletedAllocationListRoute />,
                },
                {
                  path: "performance",
                  element: <OperationsPerformanceRoute />,
                },
                {
                  path: "configurations",
                  element: <ConfigurationListRoute />,
                },
                {
                  path: "configurations/:kind/:name/:version",
                  element: <ConfigurationDetailRoute />,
                },
                {
                  path: "credentials",
                  element: <CredentialListRoute />,
                },
                {
                  path: "credentials/:credentialId",
                  element: <CredentialDetailRoute />,
                },
                {
                  path: "settings",
                  element: <OperationsSettingsRoute />,
                },
              ],
            },
          ],
        },
      ],
    },
    { path: "*", element: <NotFoundRoute /> },
  ];
}

export function createApplicationRouter() {
  return createBrowserRouter(applicationRoutes());
}
