import { useInfiniteQuery, type QueryKey } from "@tanstack/react-query";
import { useMemo } from "react";

import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import type { components } from "../../api/generated/public";
import { listRuntimeLabels } from "../../api/operations";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { listConfigurations, listCredentials } from "../../api/workflows";
import { nextPageCursor } from "../../app/pagination";
import { artifactOptionKey } from "../../run-drafts/validation";

interface CursorPage<T> {
  items: T[];
  page: components["schemas"]["PageInfo"];
}

/**
 * The pages of a picker list read so far; LoadMoreButton reads the next
 * one. Disabled lists read nothing until `enabled`.
 */
function usePickerPages<T>(
  queryKey: QueryKey,
  list: (request: { cursor?: string }) => Promise<CursorPage<T>>,
  enabled = true,
) {
  const query = useInfiniteQuery({
    queryKey,
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      list(pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: (page) => nextPageCursor(page.page),
    enabled,
  });
  const items = useMemo(
    () => query.data?.pages.flatMap((page) => page.items) ?? [],
    [query.data],
  );
  return { query, items };
}

/**
 * The choices of a Run form: the input files of the library or the
 * Project's materials (with the revisions this draft already knows), and,
 * once their disclosure opens, the published model policies, Gateways,
 * credentials and Runtime labels.
 */
export function useRunFormInventories({
  projectId,
  knownArtifacts,
  executionOptionsOpen,
  runtimeOptionsOpen,
}: {
  projectId: string | undefined;
  knownArtifacts: readonly ArtifactMetadata[];
  executionOptionsOpen: boolean;
  runtimeOptionsOpen: boolean;
}) {
  const api = usePublicAPI();
  const artifactPages = usePickerPages(
    projectId === undefined
      ? queryKeys.artifacts.picker
      : queryKeys.projects.artifacts.picker(projectId),
    (request) =>
      projectId === undefined
        ? listArtifacts(api, request)
        : listProjectArtifacts(api, { projectId, ...request }),
  );
  const modelPolicies = usePickerPages(
    queryKeys.configurations.infinitePicker("model-policies"),
    (request) => listConfigurations(api, "model-policies", request),
    executionOptionsOpen,
  );
  const gateways = usePickerPages(
    queryKeys.configurations.infinitePicker("llm-gateways"),
    (request) => listConfigurations(api, "llm-gateways", request),
    executionOptionsOpen,
  );
  const credentials = usePickerPages(
    queryKeys.credentials.picker,
    (request) => listCredentials(api, request),
    executionOptionsOpen,
  );
  const runtimeLabels = usePickerPages(
    queryKeys.operations.runtimeLabels.picker,
    (request) => listRuntimeLabels(api, request),
    runtimeOptionsOpen,
  );

  // A revision this draft selected stays a choice when no loaded page has it.
  const artifacts = useMemo(
    () =>
      Array.from(
        new Map(
          [...artifactPages.items, ...knownArtifacts].map((item) => [
            artifactOptionKey(item.artifact),
            item,
          ]),
        ).values(),
      ),
    [artifactPages.items, knownArtifacts],
  );
  const artifactMap = useMemo(
    () =>
      new Map(
        artifacts.map((metadata) => [
          artifactOptionKey(metadata.artifact),
          metadata,
        ]),
      ),
    [artifacts],
  );
  const explicitRuntimeLabels = useMemo(
    () =>
      runtimeLabels.items
        .filter((binding) => binding.label !== "default")
        .sort((left, right) => left.label.localeCompare(right.label)),
    [runtimeLabels.items],
  );
  const defaultRuntimeConfig = useMemo(
    () => runtimeLabels.items.find((binding) => binding.label === "default"),
    [runtimeLabels.items],
  );

  return {
    artifactInventory: artifactPages.query,
    artifacts,
    artifactMap,
    modelPolicyInventory: modelPolicies.query,
    modelPolicies: modelPolicies.items,
    gatewayInventory: gateways.query,
    gateways: gateways.items,
    credentialInventory: credentials.query,
    credentials: credentials.items,
    runtimeLabelInventory: runtimeLabels.query,
    runtimeLabels: explicitRuntimeLabels,
    defaultRuntimeConfig,
  };
}

export type RunFormInventories = ReturnType<typeof useRunFormInventories>;
