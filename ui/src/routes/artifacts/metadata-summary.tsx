import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { getProject } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata } from "../../api/artifacts";
import { formatBytes, formatTimestamp } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";
import { Button, Facts, IdChip, StatusGlyph, TechnicalDetails } from "../../ui";
import { ArtifactHistory } from "./history";
import { MaterialIcon } from "./icons";
import { formatLabel } from "./kinds";
import "./materials.css";

/** Id of the technical details that hold versions and lineage. */
const HISTORY_ID = "artifact-history";

/**
 * Whether the shown revision is the current one. Nothing is claimed before
 * the metadata has loaded.
 */
export function ArtifactRevisionLede({
  metadata,
}: {
  metadata: ArtifactMetadata | undefined;
}) {
  if (metadata === undefined) return null;
  return (
    <p
      className="lede materials-revision"
      data-current={metadata.current ? "" : undefined}
    >
      <StatusGlyph tone={metadata.current ? "done" : "idle"} size={14} />
      {metadata.current ? "Current revision" : "Historical revision"}
    </p>
  );
}

/** Format, size, creation, lock and the Git source of one revision. */
export function ArtifactFacts({
  metadata,
  scope,
}: {
  metadata: ArtifactMetadata;
  scope?: ArtifactArchiveScope;
}) {
  const api = usePublicAPI();
  const projectId = scope?.kind === "project" ? scope.id : undefined;
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId ?? ""),
    queryFn: () => getProject(api, projectId!),
    enabled: projectId !== undefined,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const git = metadata.gitSource;
  return (
    <Facts
      layout="row"
      className="materials-facts"
      items={[
        ...(scope?.kind === "project"
          ? [
              {
                term: "Project",
                value: (
                  <Link to={`/projects/${encodeURIComponent(scope.id)}`}>
                    {project.data?.name ?? scope.id}
                  </Link>
                ),
              },
            ]
          : scope?.kind === "run"
            ? [
                {
                  term: "Run",
                  value: (
                    <Link to={`/runs/${encodeURIComponent(scope.id)}`}>
                      {scope.id}
                    </Link>
                  ),
                },
              ]
            : scope?.kind === "user"
              ? [{ term: "Location", value: "My files" }]
              : []),
        {
          term: "Created",
          value: (
            <time dateTime={metadata.createdAt}>
              {formatTimestamp(metadata.createdAt)}
            </time>
          ),
        },
        {
          term: "Format · Size",
          value: `${formatLabel(metadata.mediaType)} · ${formatBytes(metadata.size)}`,
        },
        ...(git === undefined
          ? []
          : [
              {
                term: "Git repository",
                value: <span className="ui-mono">{git.repositoryUrl}</span>,
                wide: true,
              },
              {
                term: "Branch or tag",
                value: git.requestedRef ?? "Default branch",
              },
              {
                term: "Git commit",
                value: <IdChip value={git.resolvedCommit} label="Git commit" />,
              },
              ...(git.analysis === undefined
                ? []
                : [
                    {
                      term: "Source coverage",
                      value: `Sources for analysis · ${git.analysis.includedFiles} source files · ${git.analysis.excludedEntries} entries excluded. Full list: ${git.analysis.reportPath} inside the source ZIP.`,
                      wide: true,
                    },
                  ]),
              {
                term: "Imported",
                value: <RecordedTime value={git.importedAt} />,
              },
            ]),
      ]}
    />
  );
}

/**
 * Revision, media type, versions and lineage behind Technical details. The
 * history loads the first time the details open.
 */
export function ArtifactTechnicalDetails({
  scope,
  metadata,
  titleAs,
  separateVersions = false,
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
  /** Heading level of Versions and Lineage. Default "h3". */
  titleAs?: "h2" | "h3" | undefined;
  separateVersions?: boolean;
}) {
  const [versionsOpened, setVersionsOpened] = useState(false);
  const [opened, setOpened] = useState(false);
  return (
    <div id={HISTORY_ID} className="materials-tech">
      {separateVersions ? (
        <TechnicalDetails
          summary="Versions"
          onToggle={(open) => {
            if (open) setVersionsOpened(true);
          }}
        >
          {versionsOpened ? (
            <ArtifactHistory
              key={metadata.artifact.revision}
              scope={scope}
              metadata={metadata}
              titleAs={titleAs}
              section="versions"
            />
          ) : null}
        </TechnicalDetails>
      ) : null}
      <TechnicalDetails
        description="Revision, storage dates, versions and lineage."
        onToggle={(open) => {
          if (open) setOpened(true);
        }}
      >
        <Facts
          layout="row"
          className="materials-facts materials-tech-facts"
          items={[
            {
              term: "Revision",
              value: <code>{metadata.artifact.revision}</code>,
              wide: true,
            },
            { term: "Media type", value: <code>{metadata.mediaType}</code> },
            { term: "Locked", value: metadata.frozen ? "yes" : "no" },
          ]}
        />
        <Facts
          layout="row"
          className="materials-facts materials-tech-facts"
          items={(
            [
              [
                "Storage scope created",
                metadata.storageTimestamps.scopeCreatedAt,
              ],
              [
                "Shared blob registered",
                metadata.storageTimestamps.blobCreatedAt,
              ],
              [
                "Storage version created",
                metadata.storageTimestamps.versionCreatedAt,
              ],
              [
                "Current binding changed",
                metadata.storageTimestamps.bindingUpdatedAt,
              ],
            ] as const
          ).map(([term, value]) => ({
            term,
            value: (
              <time dateTime={value} title={value}>
                {formatTimestamp(value)} ·{" "}
                {Intl.DateTimeFormat().resolvedOptions().timeZone}
              </time>
            ),
          }))}
        />
        <p className="materials-storage-note">
          Shared blob and storage version dates can predate this material or its
          storage scope. Blob registration is not this upload's date or a file's
          creation time. Current binding changes include locking and unlocking;
          they do not necessarily change content, even when viewing a historical
          revision.
        </p>
        {opened ? (
          <ArtifactHistory
            key={metadata.artifact.revision}
            scope={scope}
            metadata={metadata}
            titleAs={titleAs}
            section={separateVersions ? "lineage" : undefined}
          />
        ) : null}
      </TechnicalDetails>
    </div>
  );
}

/** Opens Technical details at the version history and moves focus there. */
export function ArtifactVersionsButton() {
  return (
    <Button
      size="sm"
      variant="ghost"
      icon={<MaterialIcon name="history" />}
      onClick={() => {
        const node = document.getElementById(HISTORY_ID);
        const details = node?.querySelector("details");
        if (node === null || details === null || details === undefined) return;
        details.open = true;
        const reduceMotion =
          typeof window.matchMedia === "function" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        node.scrollIntoView?.({
          block: "start",
          behavior: reduceMotion ? "auto" : "smooth",
        });
        details.querySelector("summary")?.focus({ preventScroll: true });
      }}
    >
      Versions
    </Button>
  );
}
