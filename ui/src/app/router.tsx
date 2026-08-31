import { createBrowserRouter, Navigate, type RouteObject } from "react-router";

import { AuthenticatedRoute } from "../routes/guard";
import { LoginRoute } from "../routes/login";
import { ArtifactDetailRoute } from "../routes/artifacts/detail";
import { ArtifactListRoute } from "../routes/artifacts/list";
import { AcceptedRunRoute } from "../routes/runs/accepted";
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
            { path: "/runs", element: <PlaceholderRoute kind="runs" /> },
            { path: "/runs/:runId", element: <AcceptedRunRoute /> },
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
