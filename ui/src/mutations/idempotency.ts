const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const KEY_SCOPE_PATTERN = /^[a-z][a-z0-9-]{0,31}$/;

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

export function canonicalMutationRequest(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

export function createMutationIdempotencyKey(scope: string): string {
  if (!KEY_SCOPE_PATTERN.test(scope)) {
    throw new TypeError("Idempotency key scope is invalid");
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  const random = Array.from(bytes, (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
  return `${scope}-ui-${random}`;
}

export class MutationDraftKeyring<T> {
  readonly #generate: () => string;
  #canonical: string | undefined;
  #key: string | undefined;

  constructor(
    scope: string,
    generate: () => string = () => createMutationIdempotencyKey(scope),
  ) {
    this.#generate = generate;
  }

  keyFor(request: T): string {
    const canonical = canonicalMutationRequest(request);
    if (canonical === this.#canonical && this.#key !== undefined) {
      return this.#key;
    }
    const key = this.#generate();
    if (!IDEMPOTENCY_KEY_PATTERN.test(key)) {
      throw new TypeError("Generated mutation idempotency key is invalid");
    }
    this.#canonical = canonical;
    this.#key = key;
    return key;
  }

  matches(request: T): boolean {
    return canonicalMutationRequest(request) === this.#canonical;
  }
}
