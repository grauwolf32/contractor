import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  downloadRunArtifact,
  getRunArtifactMetadata,
  previewRunArtifact,
} from "../../api/runs";
import { saveBlob } from "../../app/download";

/** Explicit, exact-revision output actions shared by Run and Project results.
 * Callers key the result by revision so an already-open preview cannot follow
 * a newly published output without another user action.
 */
export function useRunArtifactActions(
  runId: string,
  artifact: ArtifactMetadata["artifact"] | undefined,
) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [requested, setRequested] = useState(false);
  const metadataKey = queryKeys.runs.artifactMetadata(
    runId,
    artifact?.namespace ?? "missing",
    artifact?.name ?? "missing",
    artifact?.revision,
  );
  function readMetadata() {
    if (artifact === undefined) throw new Error("Run output is unavailable");
    return getRunArtifactMetadata(api, { runId, ...artifact });
  }
  const metadata = useQuery({
    queryKey: metadataKey,
    queryFn: readMetadata,
    enabled: requested && artifact !== undefined,
  });
  const download = useMutation({
    mutationFn: async () => {
      const exact = await queryClient.fetchQuery({
        queryKey: metadataKey,
        queryFn: readMetadata,
      });
      return downloadRunArtifact(api, runId, exact);
    },
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });
  return {
    requested,
    requestPreview: () => setRequested(true),
    metadata,
    download,
    loadPreview: (data: ArtifactMetadata) =>
      previewRunArtifact(api, runId, data),
  };
}
