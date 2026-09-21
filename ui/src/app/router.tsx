import type { ComponentType } from "react";
import { createBrowserRouter, type RouteObject } from "react-router";

import { AuthenticatedRoute } from "../routes/guard";
import { LoginRoute } from "../routes/login";
import { NotFoundRoute, RouteChunkLoading } from "../routes/placeholders";
import { ApplicationShell } from "./shell";

type RouteModule = Record<string, unknown>;

/**
 * Route-level code splitting: each route module is a separate chunk that the
 * router loads on first navigation, so the eager bundle carries only the
 * shell, the guard and the shared API layer.
 */
function lazyRoute<M extends RouteModule, K extends keyof M>(
  load: () => Promise<M>,
  name: K,
): NonNullable<RouteObject["lazy"]> {
  return {
    Component: async () => (await load())[name] as ComponentType,
  };
}

const projectSections = () => import("../routes/projects/sections");
const projectFindings = () => import("../routes/projects/audits/findings");

function projectSection(
  section:
    "overview" | "artifacts" | "workflows" | "runs" | "audits" | "settings",
): NonNullable<RouteObject["lazy"]> {
  return {
    element: async () => {
      const { ProjectSectionRoute } = await projectSections();
      return <ProjectSectionRoute section={section} />;
    },
  };
}

export function applicationRoutes(): RouteObject[] {
  return [
    { path: "/login", element: <LoginRoute /> },
    {
      element: <AuthenticatedRoute />,
      children: [
        {
          element: <ApplicationShell />,
          hydrateFallbackElement: <RouteChunkLoading />,
          children: [
            {
              index: true,
              lazy: lazyRoute(() => import("../routes/home"), "HomeRoute"),
            },
            {
              path: "/projects",
              lazy: lazyRoute(
                () => import("../routes/projects/list"),
                "ProjectListRoute",
              ),
            },
            {
              path: "/projects/:projectId",
              lazy: lazyRoute(
                () => import("../routes/projects/detail"),
                "ProjectDetailRoute",
              ),
              children: [
                { index: true, lazy: projectSection("overview") },
                { path: "artifacts", lazy: projectSection("artifacts") },
                { path: "workflows", lazy: projectSection("workflows") },
                { path: "runs", lazy: projectSection("runs") },
                { path: "audits", lazy: projectSection("audits") },
                {
                  path: "findings",
                  lazy: {
                    element: async () => {
                      const { ProjectFindingsRoute } = await projectFindings();
                      return <ProjectFindingsRoute embedded />;
                    },
                  },
                },
                { path: "settings", lazy: projectSection("settings") },
              ],
            },
            {
              path: "/projects/:projectId/artifacts/:namespace/:name",
              lazy: lazyRoute(
                () => import("../routes/projects/artifact-detail"),
                "ProjectArtifactDetailRoute",
              ),
            },
            {
              path: "/projects/:projectId/workflows/:name/:version/run",
              lazy: lazyRoute(
                () => import("../routes/projects/workflow-run"),
                "ProjectWorkflowRunRoute",
              ),
            },
            {
              path: "/projects/:projectId/audits/:auditId",
              lazy: lazyRoute(
                () => import("../routes/projects/audits/detail"),
                "ProjectAuditDetailRoute",
              ),
            },
            {
              path: "/projects/:projectId/audits/:auditId/:section",
              lazy: lazyRoute(
                () => import("../routes/projects/audits/detail"),
                "ProjectAuditDetailRoute",
              ),
            },
            {
              path: "/evals",
              lazy: lazyRoute(
                () => import("../routes/evals/list"),
                "EvalListRoute",
              ),
            },
            {
              path: "/evals/legacy",
              lazy: lazyRoute(
                () => import("../routes/projects/list"),
                "EvaluationListRoute",
              ),
            },
            {
              path: "/evals/new",
              lazy: lazyRoute(
                () => import("../routes/evals/setup"),
                "EvalNewRoute",
              ),
            },
            {
              path: "/evals/datasets",
              lazy: lazyRoute(
                () => import("../routes/evals/datasets"),
                "EvalDatasetsRoute",
              ),
            },
            {
              path: "/evals/experiments/:experimentId",
              lazy: lazyRoute(
                () => import("../routes/evals/detail"),
                "EvalDetailRoute",
              ),
            },
            {
              path: "/evals/experiments/:experimentId/:section",
              lazy: lazyRoute(
                () => import("../routes/evals/detail"),
                "EvalDetailRoute",
              ),
            },
            {
              path: "/evals/experiments/:experimentId/pairs/:pairId",
              lazy: lazyRoute(
                () => import("../routes/evals/pair"),
                "EvalPairRoute",
              ),
            },
            {
              path: "/evals/:projectId",
              lazy: lazyRoute(
                () => import("../routes/projects/detail"),
                "EvaluationDetailRoute",
              ),
            },
            {
              path: "/evals/:projectId/artifacts/:namespace/:name",
              lazy: lazyRoute(
                () => import("../routes/projects/artifact-detail"),
                "EvaluationArtifactDetailRoute",
              ),
            },
            {
              path: "/artifacts",
              lazy: lazyRoute(
                () => import("../routes/artifacts/list"),
                "ArtifactListRoute",
              ),
            },
            {
              path: "/artifacts/:namespace/:name",
              lazy: lazyRoute(
                () => import("../routes/artifacts/detail"),
                "ArtifactDetailRoute",
              ),
            },
            {
              path: "/runs",
              lazy: lazyRoute(() => import("../routes/runs"), "RunsRoute"),
              children: [
                {
                  path: "configuration",
                  lazy: lazyRoute(
                    () => import("../routes/runs/configuration"),
                    "RunConfigurationLayout",
                  ),
                  children: [
                    {
                      index: true,
                      lazy: lazyRoute(
                        () => import("../routes/operations/runtime-configs"),
                        "RuntimeConfigurationRoute",
                      ),
                    },
                    {
                      path: ":name/:version",
                      lazy: lazyRoute(
                        () =>
                          import("../routes/operations/runtime-configs/detail"),
                        "RuntimeConfigDetailRoute",
                      ),
                    },
                  ],
                },
              ],
            },
            {
              path: "/runs/:runId",
              lazy: lazyRoute(
                () => import("../routes/runs/detail"),
                "RunDetailRoute",
              ),
            },
            {
              path: "/runs/:runId/artifacts/:namespace/:name",
              lazy: lazyRoute(
                () => import("../routes/runs/artifacts"),
                "RunArtifactDetailRoute",
              ),
            },
            {
              path: "/catalog",
              lazy: lazyRoute(
                () => import("../routes/catalog/layout"),
                "CatalogLayoutRoute",
              ),
              children: [
                {
                  index: true,
                  lazy: lazyRoute(
                    () => import("../routes/catalog/layout"),
                    "CatalogIndexRedirect",
                  ),
                },
                {
                  path: "workflows",
                  lazy: lazyRoute(
                    () => import("../routes/workflows/list"),
                    "WorkflowListRoute",
                  ),
                },
                {
                  path: "workflows/:name/:version",
                  lazy: lazyRoute(
                    () => import("../routes/workflows/detail"),
                    "WorkflowDetailRoute",
                  ),
                },
                {
                  path: "agents",
                  lazy: lazyRoute(
                    () => import("../routes/catalog/agents"),
                    "AgentListRoute",
                  ),
                },
                {
                  path: "audit-presets",
                  lazy: lazyRoute(
                    () => import("../routes/catalog/audit-presets"),
                    "AuditPresetListRoute",
                  ),
                },
                {
                  path: "audit-presets/:name/:version",
                  lazy: lazyRoute(
                    () => import("../routes/catalog/audit-preset-detail"),
                    "AuditPresetDetailRoute",
                  ),
                },
                {
                  path: "agents/:name/:version",
                  lazy: lazyRoute(
                    () => import("../routes/catalog/agent-detail"),
                    "AgentDetailRoute",
                  ),
                },
                {
                  path: "skills",
                  lazy: lazyRoute(
                    () => import("../routes/skills"),
                    "SkillsRoute",
                  ),
                },
              ],
            },
            {
              path: "/operations",
              lazy: lazyRoute(
                () => import("../routes/operations/layout"),
                "OperationsLayoutRoute",
              ),
              children: [
                {
                  index: true,
                  lazy: lazyRoute(
                    () => import("../routes/operations/overview"),
                    "OperationsOverviewRoute",
                  ),
                },
                {
                  path: "runtime-agents",
                  lazy: lazyRoute(
                    () => import("../routes/operations/runtime-agents"),
                    "RuntimeAgentListRoute",
                  ),
                },
                {
                  path: "allocations",
                  lazy: lazyRoute(
                    () => import("../routes/operations/allocations"),
                    "AllocationListRoute",
                  ),
                },
                {
                  path: "allocations/completed",
                  lazy: lazyRoute(
                    () => import("../routes/operations/allocations/completed"),
                    "CompletedAllocationListRoute",
                  ),
                },
                {
                  path: "performance",
                  lazy: lazyRoute(
                    () => import("../routes/operations/performance"),
                    "OperationsPerformanceRoute",
                  ),
                },
                {
                  path: "configurations",
                  lazy: lazyRoute(
                    () => import("../routes/operations/llm-configurations"),
                    "ConfigurationListRoute",
                  ),
                },
                {
                  path: "configurations/:kind/:name/:version",
                  lazy: lazyRoute(
                    () =>
                      import("../routes/operations/llm-configurations/detail"),
                    "ConfigurationDetailRoute",
                  ),
                },
                {
                  path: "credentials",
                  lazy: lazyRoute(
                    () => import("../routes/operations/credentials"),
                    "CredentialListRoute",
                  ),
                },
                {
                  path: "credentials/:credentialId",
                  lazy: lazyRoute(
                    () => import("../routes/operations/credentials/detail"),
                    "CredentialDetailRoute",
                  ),
                },
                {
                  path: "settings",
                  lazy: lazyRoute(
                    () => import("../routes/operations/settings"),
                    "OperationsSettingsRoute",
                  ),
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
