import type { PublicAPI } from "../api/client";
import {
  createRuntimeCredential,
  publishRuntimeConfig,
  putRuntimeLabel,
  type CreateRuntimeCredentialRequest,
  type RuntimeConfigAuthorDocument,
  type RuntimeLabelBinding,
} from "../api/operations";
import {
  canonicalMutationRequest,
  MutationDraftKeyring,
} from "../mutations/idempotency";

export async function privateMutationKey(
  keys: MutationDraftKeyring<{ digest: string }>,
  request: unknown,
): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalMutationRequest(request));
  try {
    const hash = await crypto.subtle.digest("SHA-256", bytes);
    const digest = Array.from(new Uint8Array(hash), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    return keys.keyFor({ digest });
  } finally {
    bytes.fill(0);
  }
}

// Only safe identities and request fingerprints survive dialog navigation.
// Neither RunDraftState nor React Query receives endpoint/header form state.
export class ToolsetConnectionCreation {
  label = "";
  credentialId: string | undefined;
  config: RuntimeLabelBinding["config"] | undefined;
  binding: RuntimeLabelBinding | undefined;
  step: "credential" | "configuration" | "label" | "complete" = "configuration";
  #revision = 0;
  #listeners = new Set<() => void>();
  #flight: Promise<RuntimeLabelBinding> | undefined;
  #credentialKeys = new MutationDraftKeyring<{ digest: string }>(
    "mcp-credential",
  );
  #configKeys = new MutationDraftKeyring<{ digest: string }>("mcp-config");
  #labelKeys = new MutationDraftKeyring<{
    label: string;
    config: RuntimeLabelBinding["config"];
  }>("mcp-label");

  get pending(): boolean {
    return this.#flight !== undefined;
  }
  snapshot = (): number => this.#revision;
  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };
  #changed(): void {
    this.#revision++;
    for (const listener of this.#listeners) listener();
  }

  run(
    api: PublicAPI,
    options: {
      label: string;
      document?: RuntimeConfigAuthorDocument;
      credential?: CreateRuntimeCredentialRequest;
    },
  ): Promise<RuntimeLabelBinding> {
    if (this.#flight !== undefined) return this.#flight;
    this.label = options.label;
    // Promise dispatch starts after #flight is installed, including observers.
    this.#flight = Promise.resolve().then(async () => {
      try {
        if (this.binding !== undefined) return this.binding;
        if (this.config === undefined) {
          if (options.document === undefined)
            throw new Error(
              "Re-enter the connection settings to continue; secret values cannot be recovered.",
            );
          if (
            options.credential !== undefined &&
            this.credentialId === undefined
          ) {
            this.step = "credential";
            this.#changed();
            const key = await privateMutationKey(
              this.#credentialKeys,
              options.credential,
            );
            try {
              const metadata = await createRuntimeCredential(
                api,
                options.credential,
                key,
              );
              this.credentialId = metadata.credentialId;
            } finally {
              if ("headers" in options.credential.material)
                for (const name of Object.keys(
                  options.credential.material.headers,
                ))
                  delete options.credential.material.headers[name];
            }
          }
          this.step = "configuration";
          this.#changed();
          const document = structuredClone(options.document);
          const mcp = document.spec.worker?.toolsets?.["mcp@1"];
          if (mcp == null)
            throw new Error("MCP connection settings are required.");
          if (this.credentialId !== undefined)
            mcp.credential = this.credentialId;
          const key = await privateMutationKey(this.#configKeys, document);
          this.config = (await publishRuntimeConfig(api, document, key)).ref;
        }
        this.step = "label";
        this.#changed();
        const key = this.#labelKeys.keyFor({
          label: this.label,
          config: this.config,
        });
        this.binding = await putRuntimeLabel(api, this.label, this.config, key);
        this.step = "complete";
        return this.binding;
      } finally {
        // API requests hold material only for the lifetime of that operation.
        if (
          options.credential !== undefined &&
          "headers" in options.credential.material
        ) {
          for (const key of Object.keys(options.credential.material.headers))
            delete options.credential.material.headers[key];
        }
        this.#flight = undefined;
        this.#changed();
      }
    });
    this.#changed();
    return this.#flight;
  }
}

export type ToolsetCoverageState = {
  ref: string;
  state: "configured" | "missing" | "cleared" | "conflict" | "unavailable";
  sources: string[];
};

export function resolveToolsetCoverage(
  refs: readonly string[],
  selected: readonly string[],
  bindings: readonly RuntimeLabelBinding[],
): ToolsetCoverageState[] {
  const byLabel = new Map(bindings.map((binding) => [binding.label, binding]));
  return refs.map((ref) => {
    if (["default", ...selected].some((label) => !byLabel.has(label)))
      return {
        ref,
        state: "unavailable",
        sources: ["default", ...selected].filter(
          (label) => !byLabel.has(label),
        ),
      };
    const overrides = [...selected].sort().flatMap((label) => {
      const value = byLabel
        .get(label)
        ?.toolsetConfigurationCoverage?.find((item) => item.ref === ref);
      return value === undefined ? [] : [{ label, value }];
    });
    if (overrides.length) {
      const identity = (row: (typeof overrides)[number]) =>
        row.value.state === "cleared" ? "cleared" : row.value.valueDigest;
      if (new Set(overrides.map(identity)).size > 1)
        return {
          ref,
          state: "conflict",
          sources: overrides.map((row) => row.label),
        };
      return {
        ref,
        state: overrides[0]!.value.state,
        sources: overrides.map((row) => row.label),
      };
    }
    const inherited = byLabel
      .get("default")
      ?.toolsetConfigurationCoverage?.find((item) => item.ref === ref);
    return {
      ref,
      state: inherited?.state ?? "missing",
      sources: inherited === undefined ? [] : ["default"],
    };
  });
}
