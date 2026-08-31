import { createBrowserRouter, Navigate, type RouteObject } from "react-router";

import { AuthenticatedRoute } from "../routes/guard";
import { LoginRoute } from "../routes/login";
import { ArtifactDetailRoute } from "../routes/artifacts/detail";
import { ArtifactListRoute } from "../routes/artifacts/list";
import { RunArtifactDetailRoute } from "../routes/runs/artifacts";
import { RunDetailRoute } from "../routes/runs/detail";
import { RunListRoute } from "../routes/runs/list";
import { WorkflowDetailRoute } from "../routes/workflows/detail";
import { WorkflowListRoute } from "../routes/workflows/list";
import { NotFoundRoute, PlaceholderRoute } from "../routes/placeholders";
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
            { index: true, element: <Navigate to="/workflows" replace /> },
            {
              path: "/workflows",
              element: <WorkflowListRoute />,
            },
            {
              path: "/workflows/:name/:version",
              element: <WorkflowDetailRoute />,
            },
            {
              path: "/artifacts",
              element: <ArtifactListRoute />,
            },
            {
              path: "/artifacts/:namespace/:name",
              element: <ArtifactDetailRoute />,
            },
            { path: "/runs", element: <RunListRoute /> },
            { path: "/runs/:runId", element: <RunDetailRoute /> },
            {
              path: "/runs/:runId/artifacts/:namespace/:name",
              element: <RunArtifactDetailRoute />,
            },
            {
              path: "/operations",
              element: <PlaceholderRoute kind="operations" />,
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
