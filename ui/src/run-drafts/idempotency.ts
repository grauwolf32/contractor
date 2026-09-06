import type { CreateRunRequest } from "../api/workflows";

const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalValue);
  }
  if (typeof value === "object" && value !== null) {
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      const child = (value as Record<string, unknown>)[key];
      if (child !== undefined) {
        result[key] = canonicalValue(child);
      }
    }
    return result;
  }
  return value;
}

export function canonicalRunRequest(request: CreateRunRequest): string {
  return JSON.stringify(canonicalValue(request));
}

export function createRunIdempotencyKey(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  const random = Array.from(bytes, (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  return `run-ui-${random}`;
}

export class RunDraftKeyring {
  readonly #generate: () => string;
  #canonical: string | undefined;
  #key: string | undefined;

  constructor(generate: () => string = createRunIdempotencyKey) {
    this.#generate = generate;
  }

  keyFor(request: CreateRunRequest, endpointIdentity = "standalone"): string {
    const canonical = `${endpointIdentity}\u0000${canonicalRunRequest(request)}`;
    if (canonical === this.#canonical && this.#key !== undefined) {
      return this.#key;
    }
    const key = this.#generate();
    if (!IDEMPOTENCY_KEY_PATTERN.test(key)) {
      throw new TypeError("Generated Run idempotency key is invalid");
    }
    this.#canonical = canonical;
    this.#key = key;
    return key;
  }

  matches(request: CreateRunRequest, endpointIdentity = "standalone"): boolean {
    return (
      `${endpointIdentity}\u0000${canonicalRunRequest(request)}` ===
      this.#canonical
    );
  }

  hasSubmission(): boolean {
    return this.#canonical !== undefined && this.#key !== undefined;
  }
}
