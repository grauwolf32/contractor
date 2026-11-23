import type { CreateRunRequest } from "../api/workflows";
import {
  createMutationIdempotencyKey,
  MutationDraftKeyring,
} from "../mutations/idempotency";

/**
 * Keeps one idempotency key per exact Run request and creation endpoint, so a
 * key is never reused across the standalone and Project endpoints.
 */
export class RunDraftKeyring {
  readonly #keyring: MutationDraftKeyring<{
    endpoint: string;
    request: CreateRunRequest;
  }>;

  constructor(
    generate: () => string = () => createMutationIdempotencyKey("run"),
  ) {
    this.#keyring = new MutationDraftKeyring("run", generate);
  }

  keyFor(request: CreateRunRequest, endpointIdentity = "standalone"): string {
    return this.#keyring.keyFor({ endpoint: endpointIdentity, request });
  }

  matches(request: CreateRunRequest, endpointIdentity = "standalone"): boolean {
    return this.#keyring.matches({ endpoint: endpointIdentity, request });
  }

  hasSubmission(): boolean {
    return this.#keyring.hasSubmission();
  }
}
