import { useInfiniteQuery, type QueryKey } from "@tanstack/react-query";
import { useMemo } from "react";

import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import type { components } from "../../api/generated/public";
import { listRuntimeLabels } from "../../api/operations";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import type { WorkflowResource } from "../../api/workflows";
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
 * The choices of a Run form: each input uses the library or the
 * Project's materials (with the revisions this draft already knows), and,
 * once their disclosure opens, Runtime labels.
 */
export function useRunFormInventories({
  projectId,
  inputs,
  knownArtifacts,
  runtimeOptionsOpen,
}: {
  projectId: string | undefined;
  inputs: WorkflowResource["inputs"];
  knownArtifacts: Readonly<Record<string, readonly ArtifactMetadata[]>>;
  runtimeOptionsOpen: boolean;
}) {
  const api = usePublicAPI();
  const hasProjectInputs =
    projectId !== undefined &&
    Object.values(inputs).some((slot) => slot.sourceScope !== "user");
  const hasUserInputs = Object.values(inputs).some(
    (slot) => projectId === undefined || slot.sourceScope === "user",
  );
  const userArtifacts = usePickerPages(
    queryKeys.artifacts.picker,
    (request) => listArtifacts(api, request),
    hasUserInputs,
  );
  const projectArtifacts = usePickerPages(
    projectId === undefined
      ? queryKeys.artifacts.picker
      : queryKeys.projects.artifacts.picker(projectId),
    (request) =>
      projectId === undefined
        ? listArtifacts(api, request)
        : listProjectArtifacts(api, { projectId, ...request }),
    hasProjectInputs,
  );
  const runtimeLabels = usePickerPages(
    queryKeys.operations.runtimeLabels.picker,
    (request) => listRuntimeLabels(api, request),
    runtimeOptionsOpen,
  );

  // Retained metadata belongs to the input's scope, even if another scope
  // contains the same exact ref key. A missing loaded page does not drop it.
  const inputInventories = useMemo(() => {
    const knownUser: ArtifactMetadata[] = [];
    const knownProject: ArtifactMetadata[] = [];
    for (const [name, slot] of Object.entries(inputs)) {
      const retained =
        projectId === undefined || slot.sourceScope === "user"
          ? knownUser
          : knownProject;
      retained.push(...(knownArtifacts[name] ?? []));
    }
    return Object.fromEntries(
      Object.entries(inputs).map(([name, slot]) => {
        const inputProjectId =
          slot.sourceScope === "user" ? undefined : projectId;
        const items =
          inputProjectId === undefined
            ? userArtifacts.items
            : projectArtifacts.items;
        const retained =
          inputProjectId === undefined ? knownUser : knownProject;
        const artifactMap = new Map(
          [...items, ...retained].map((metadata) => [
            artifactOptionKey(metadata.artifact),
            metadata,
          ]),
        );
        return [
          name,
          {
            projectId: inputProjectId,
            artifactInventory:
              inputProjectId === undefined
                ? userArtifacts.query
                : projectArtifacts.query,
            artifacts: Array.from(artifactMap.values()),
            artifactMap,
          },
        ];
      }),
    );
  }, [
    inputs,
    projectId,
    userArtifacts.items,
    userArtifacts.query,
    projectArtifacts.items,
    projectArtifacts.query,
    knownArtifacts,
  ]);
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
    inputs: inputInventories,
    artifactInventories: [
      ...(hasProjectInputs
        ? [{ projectId, query: projectArtifacts.query }]
        : []),
      ...(hasUserInputs
        ? [{ projectId: undefined, query: userArtifacts.query }]
        : []),
    ],
    artifactsPending:
      (hasProjectInputs && projectArtifacts.query.isPending) ||
      (hasUserInputs && userArtifacts.query.isPending),
    runtimeLabelInventory: runtimeLabels.query,
    runtimeLabels: explicitRuntimeLabels,
    defaultRuntimeConfig,
  };
}

export type RunFormInventories = ReturnType<typeof useRunFormInventories>;
