import type { ExactArtifactRef } from "../../../api/artifacts";
import type { PublicAPI } from "../../../api/client";
import {
  downloadProjectArtifact,
  getProjectArtifactMetadata,
  writeProjectArtifact,
} from "../../../api/project-artifacts";

async function contentHash(data: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

/** Retain exact authored inputs using ordinary Project Artifact create-only CAS. */
export async function retainAuthoredMaterial(
  api: PublicAPI,
  projectId: string,
  namespace: string,
  name: string,
  payload: Blob,
  ensureCurrent: () => void,
): Promise<ExactArtifactRef> {
  const bytes = await payload.arrayBuffer();
  const hash = await contentHash(bytes);
  ensureCurrent();
  const identity = { projectId, namespace, name: `${name}.${hash}` };
  try {
    const result = await writeProjectArtifact(api, {
      ...identity,
      payload,
      mediaType: payload.type,
    });
    ensureCurrent();
    return result.artifact;
  } catch (reason) {
    ensureCurrent();
    // A create-only write may have committed before its answer was lost.
    try {
      const existing = await getProjectArtifactMetadata(api, identity);
      ensureCurrent();
      if (
        existing.size === bytes.byteLength &&
        existing.mediaType === payload.type
      ) {
        const retained = await downloadProjectArtifact(
          api,
          projectId,
          existing,
        );
        const retainedHash = await contentHash(
          await retained.blob.arrayBuffer(),
        );
        ensureCurrent();
        if (retainedHash === hash) return existing.artifact;
      }
    } catch {
      // Keep the original write failure; never overwrite a different binding.
    }
    throw reason;
  }
}
