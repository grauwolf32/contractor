// The Evals pages under test: the application over the shared Eval fixture
// (src/test/evals-fixture.ts), whose answers the shared fake Server checks
// against the contract.
import { webcrypto } from "node:crypto";
import type { QueryClient } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { StrictMode } from "react";
import { createMemoryRouter } from "react-router";
import { vi } from "vitest";

import type { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import * as queryClients from "../../app/query-client";
import { applicationRoutes } from "../../app/router";
import type { createEvalFixture } from "../../test/evals-fixture";
import { fakeServer } from "../../test/fake-server";
import { testAPI } from "../../test/render";
import { MemoryStorage } from "../../test/storage";

type EvalFixture = ReturnType<typeof createEvalFixture>;

/** Empty browser storage and Web Crypto, as Eval command recovery needs. */
export function stubEvalBrowser(): void {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
}

/** A signed-in PublicAPI over `fixture`; it reads `fixture.fetch` per request. */
export function evalAPI(fixture: EvalFixture): PublicAPI {
  return testAPI(fakeServer({}, (request) => fixture.fetch(request))).api;
}

/**
 * The application at `path` over `fixture`. With `cache`, the application
 * uses that query client, so a test can seed or watch it; `strict` renders
 * in StrictMode.
 */
export function renderEvals(
  fixture: EvalFixture,
  path: string,
  { cache, strict = false }: { cache?: QueryClient; strict?: boolean } = {},
) {
  if (cache !== undefined)
    vi.spyOn(queryClients, "createApplicationQueryClient").mockReturnValue(
      cache,
    );
  const api = evalAPI(fixture);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const app = <Application api={api} publicAPI={api} router={router} />;
  return {
    ...render(strict ? <StrictMode>{app}</StrictMode> : app),
    router,
  };
}
