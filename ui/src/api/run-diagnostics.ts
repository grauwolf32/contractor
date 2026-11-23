import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";
import { RUN_ID_PATTERN } from "./runs";

type RuntimeDiagnosticPage = components["schemas"]["RuntimeDiagnosticPage"];

function exactUnsigned(value: string, maximum: bigint): boolean {
  return (
    /^(0|[1-9][0-9]*)$/.test(value) &&
    value.length <= 20 &&
    BigInt(value) <= maximum
  );
}

export async function getRuntimeDiagnostics(
  api: PublicAPI,
  runId: string,
  stageExecutionId: string,
  cursor?: string,
  signal?: AbortSignal,
): Promise<RuntimeDiagnosticPage> {
  if (!RUN_ID_PATTERN.test(runId) || !RUN_ID_PATTERN.test(stageExecutionId)) {
    throw new TypeError("Run or Stage execution ID is invalid");
  }
  const result = await api.request((client) =>
    client.GET(
      "/v1/runs/{runId}/attempts/{stageExecutionId}/runtime-diagnostics",
      {
        params: {
          path: { runId, stageExecutionId },
          query: { limit: 5, ...(cursor === undefined ? {} : { cursor }) },
        },
        ...(signal === undefined ? {} : { signal }),
      },
    ),
  );
  const value = requireData(result);
  const valid = value.items.every(
    (item) =>
      (item.stopReason === undefined ||
        new TextEncoder().encode(item.stopReason).byteLength <= 4096) &&
      (item.durationMs === undefined ||
        exactUnsigned(item.durationMs, 9223372036854775807n)) &&
      item.adapters.every(
        (adapter) =>
          exactUnsigned(adapter.operations, 18446744073709551615n) &&
          exactUnsigned(adapter.failedOperations, 18446744073709551615n) &&
          BigInt(adapter.failedOperations) <= BigInt(adapter.operations) &&
          Object.values(adapter.droppedSpans ?? {}).every((count) =>
            exactUnsigned(count, 18446744073709551615n),
          ),
      ),
  );
  if (
    !valid ||
    value.page.hasMore !== (value.page.nextCursor !== undefined) ||
    (value.status !== "available" && value.items.length !== 0)
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Runtime diagnostics response is inconsistent",
    );
  }
  return value;
}
