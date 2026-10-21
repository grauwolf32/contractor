import { MAXIMUM_ARTIFACT_BYTES } from "./artifacts";
import type { PublicAPI } from "./client";
import { invalidAPIResponse } from "./error";

/** Read only an exact committed Audit receipt, optionally one package member. */
export async function readAuditArtifact(
  api: PublicAPI,
  auditId: string,
  key: string,
  revision: string,
  member: string | undefined,
  signal: AbortSignal,
): Promise<{ blob: Blob; mediaType: string }> {
  const query = new URLSearchParams({ key, revision });
  if (member !== undefined) query.set("member", member);
  const response = await api.fetch(
    `/v1/audits/${encodeURIComponent(auditId)}/artifacts?${query.toString()}`,
    { method: "GET", signal },
  );
  if (!response.ok) throw await api.error(response);
  const declared = response.headers.get("Content-Length");
  const mediaType = response.headers.get("Content-Type");
  if (
    declared === null ||
    !/^(0|[1-9][0-9]*)$/u.test(declared) ||
    Number(declared) > MAXIMUM_ARTIFACT_BYTES ||
    mediaType === null
  ) {
    await response.body?.cancel();
    throw invalidAPIResponse(
      response.status,
      "The saved material has invalid metadata.",
    );
  }
  const reader = response.body?.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  if (reader !== undefined) {
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > Number(declared) || size > MAXIMUM_ARTIFACT_BYTES) {
          throw invalidAPIResponse(
            response.status,
            "The saved material exceeds its recorded length.",
          );
        }
        chunks.push(new Uint8Array(value));
      }
    } catch (error) {
      await reader.cancel().catch(() => undefined);
      throw error;
    } finally {
      reader.releaseLock();
    }
  }
  if (size !== Number(declared))
    throw invalidAPIResponse(
      response.status,
      "The saved material is incomplete.",
    );
  return { blob: new Blob(chunks, { type: mediaType }), mediaType };
}
