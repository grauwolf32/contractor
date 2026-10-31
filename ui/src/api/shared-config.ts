import { sha256Hex } from "../app/digest";
import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

type Config = components["schemas"]["SharedConfig"];
export type SharedConfig = Readonly<Omit<Config, "limits">> & {
  readonly limits: Readonly<Config["limits"]>;
};

interface SnapshotRead {
  promise: Promise<SharedConfig>;
  signal: AbortSignal | undefined;
  loaded: boolean;
}
const snapshots = new WeakMap<PublicAPI, SnapshotRead>();

/** One verified, immutable policy snapshot for this Server client. */
export function getSharedConfig(
  api: PublicAPI,
  signal?: AbortSignal,
): Promise<SharedConfig> {
  const existing = snapshots.get(api);
  if (
    existing !== undefined &&
    (existing.loaded || !existing.signal?.aborted)
  ) {
    return existing.promise;
  }
  const pending = readConfig(api, signal);
  const entry = { promise: pending, signal, loaded: false };
  snapshots.set(api, entry);
  void pending.then(
    () => {
      entry.loaded = true;
    },
    () => {
      if (snapshots.get(api) === entry) snapshots.delete(api);
    },
  );
  return pending;
}

async function readConfig(
  api: PublicAPI,
  signal?: AbortSignal,
): Promise<SharedConfig> {
  const result = await api.request((client) =>
    client.GET("/v1/config", signal === undefined ? {} : { signal }),
  );
  const value = requireData(result);
  // This scalar projection is canonical JSON: keys are sorted, with positive
  // exact integers below the exponential-notation threshold (CTL-R171).
  const projection = JSON.stringify({
    limits: { artifactPayloadMaxBytes: value.limits.artifactPayloadMaxBytes },
    version: value.version,
  });
  const digest = `sha256:${await sha256Hex(new TextEncoder().encode(projection))}`;
  if (digest !== value.digest) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned inconsistent settings.",
    );
  }
  return Object.freeze({
    ...value,
    limits: Object.freeze({ ...value.limits }),
  });
}
