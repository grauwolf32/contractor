// Test helpers of the Materials, Files and Workflow tests.
// V385: needs shared artifact builders, a payload response and an
// application harness in src/test.
import { render } from "@testing-library/react";
import { StrictMode } from "react";
import { createMemoryRouter } from "react-router";

import type { components } from "../../api/generated/public";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import { conforming, NOW } from "../../test/builders";
import {
  API_VERSION_HEADERS,
  emptyPageOrNotFound,
  fakeServer,
  invalidJSON,
  type FakeServer,
  type Routes,
} from "../../test/fake-server";
import { testAPI, testSession } from "../../test/render";

type Schemas = components["schemas"];

/** Overrides in which `undefined` removes the member. */
type Overrides<T> = { [K in keyof T]?: T[K] | undefined };

function merge<T extends object>(base: T, overrides: Overrides<T>): T {
  const result: Record<string, unknown> = { ...base, ...overrides };
  for (const [key, value] of Object.entries(result)) {
    if (value === undefined) delete result[key];
  }
  return result as T;
}

/** The current revision `r1` of a 64-byte ZIP at `namespace/name`. */
export function artifactMetadata(
  namespace: string,
  name: string,
  overrides: Overrides<Schemas["ArtifactMetadata"]> = {},
): Schemas["ArtifactMetadata"] {
  return conforming(
    "ArtifactMetadata",
    merge<Schemas["ArtifactMetadata"]>(
      {
        artifact: { namespace, name, revision: "r1" },
        mediaType: "application/zip",
        size: 64,
        current: true,
        frozen: false,
        createdAt: NOW,
      },
      overrides,
    ),
  );
}

/** The write answer that stored `metadata`. */
export function artifactWrite(
  metadata: Schemas["ArtifactMetadata"],
): Schemas["ArtifactWriteResponse"] {
  return conforming("ArtifactWriteResponse", {
    artifact: metadata.artifact,
    mediaType: metadata.mediaType,
    size: metadata.size,
  });
}

/**
 * Artifact bytes as the Server serves them, in the artifact's media type.
 * A JSON artifact is no API body, so the fake Server must not check it
 * against the operation's (absent) response schema.
 */
export function payload(body: string, mediaType: string): Response {
  if (mediaType === "application/json") {
    return invalidJSON(JSON.parse(body));
  }
  return new Response(body, {
    headers: { "Content-Type": mediaType, ...API_VERSION_HEADERS },
  });
}

/**
 * The whole application at `path`, signed in, over a fake Server that
 * answers `routes` and gives every other list read an empty page.
 */
export function renderApplication(
  path: string | { pathname: string; search?: string; state?: unknown },
  routes: Routes | FakeServer = {},
  { strict = false }: { strict?: boolean } = {},
) {
  const server =
    "fetch" in routes ? routes : fakeServer(routes, emptyPageOrNotFound);
  const { api } = testAPI(server);
  const session = testSession(["user", "operations"]);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const application = (
    <Application
      api={{
        getSession: async () => session,
        login: async () => session,
        logout: async () => undefined,
      }}
      publicAPI={api}
      router={router}
    />
  );
  return {
    ...render(strict ? <StrictMode>{application}</StrictMode> : application),
    router,
    server,
    requests: server.requests,
    sent: server.sent,
  };
}
