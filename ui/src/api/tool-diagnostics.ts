import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import { RUN_ID_PATTERN } from "./runs";

/** Arguments remain text; parsing them would round valid large JSON numbers. */
export async function getToolCallDiagnostics(
  api: PublicAPI,
  runId: string,
  stageExecutionId: string,
  participant: "planner" | "worker",
  logicalAgent: string | undefined,
  cursor?: string,
  signal?: AbortSignal,
) {
  if (!RUN_ID_PATTERN.test(runId) || !RUN_ID_PATTERN.test(stageExecutionId)) {
    throw new TypeError("Run or Stage execution ID is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/attempts/{stageExecutionId}/tool-calls", {
      params: {
        path: { runId, stageExecutionId },
        query: {
          participant,
          ...(logicalAgent === undefined ? {} : { logicalAgent }),
          limit: 25,
          ...(cursor === undefined ? {} : { cursor }),
        },
      },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  const value = requireData(result);
  const bytes = (text: string) => new TextEncoder().encode(text).byteLength;
  const metric = (text: string | null) =>
    text === null ||
    (/^(0|[1-9][0-9]*)$/.test(text) &&
      text.length <= 19 &&
      BigInt(text) <= 9223372036854775807n);
  if (
    value.participant !== participant ||
    value.logicalAgent !== logicalAgent ||
    value.page.hasMore !== (value.page.nextCursor !== undefined) ||
    (value.status !== "available" && value.items.length !== 0) ||
    (value.status === "available" && value.complete === undefined) ||
    !value.items.every(
      (item) =>
        metric(item.durationMs) &&
        metric(item.resultSizeBytes) &&
        bytes(item.tool) <= 128 &&
        (item.argumentsJson === undefined ||
          bytes(item.argumentsJson) <= 4096) &&
        (item.error === undefined || bytes(item.error.message) <= 4096),
    )
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Tool-call diagnostics response is inconsistent",
    );
  }
  return value;
}
