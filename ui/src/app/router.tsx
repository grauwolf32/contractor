import { createBrowserRouter, Navigate, type RouteObject } from "react-router";

import { AuthenticatedRoute } from "../routes/guard";
import { LoginRoute } from "../routes/login";
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
              element: <PlaceholderRoute kind="workflows" />,
            },
            {
              path: "/artifacts",
              element: <PlaceholderRoute kind="artifacts" />,
            },
            { path: "/runs", element: <PlaceholderRoute kind="runs" /> },
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
