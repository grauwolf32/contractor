// The Evals pages under test: the shared Eval fixture
// (src/test/evals-fixture.ts) as a fake Server, whose answers it checks
// against the contract.
import { webcrypto } from "node:crypto";
import { vi } from "vitest";

import type { createEvalFixture } from "../../test/evals-fixture";
import { fakeServer, type FakeServer } from "../../test/fake-server";
import { MemoryStorage } from "../../test/storage";

type EvalFixture = ReturnType<typeof createEvalFixture>;

/** Empty browser storage and Web Crypto, as Eval command recovery needs. */
export function stubEvalBrowser(): void {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
}

/**
 * A fake Server over `fixture`, which reads `fixture.fetch` per request and
 * answers the session with the fixture's own owner.
 */
export function evalServer(fixture: EvalFixture): FakeServer {
  return fakeServer(
    { "GET /v1/auth/session": ({ request }) => fixture.fetch(request) },
    (request) => fixture.fetch(request),
  );
}
