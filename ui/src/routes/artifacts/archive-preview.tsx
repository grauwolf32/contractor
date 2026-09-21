import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import {
  archiveQueryKey,
  getArtifactArchive,
  getArtifactArchiveFile,
  type ArtifactArchiveEntry,
  type ArtifactArchiveScope,
} from "../../api/artifact-archive";
import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { ErrorNotice, formatBytes } from "./common";
import { LoadedArtifactPreview } from "./loaded-preview";
import "./archive-preview.css";

function ArchiveTree({
  childrenByPath,
  parent = "",
  selected,
  onSelect,
}: {
  childrenByPath: Map<string, ArtifactArchiveEntry[]>;
  parent?: string;
  selected: string | null;
  onSelect: (path: string) => void;
}) {
  return (
    <ul className="archive-tree">
      {(childrenByPath.get(parent) ?? []).map((entry) => (
        <li key={entry.path}>
          {entry.kind === "directory" ? (
            <details
              open={selected?.startsWith(`${entry.path}/`) ? true : undefined}
            >
              <summary>
                <bdi>{entry.path.split("/").at(-1)}/</bdi>
              </summary>
              <ArchiveTree
                childrenByPath={childrenByPath}
                parent={entry.path}
                selected={selected}
                onSelect={onSelect}
              />
            </details>
          ) : (
            <button
              type="button"
              className="archive-file-button"
              aria-pressed={selected === entry.path}
              onClick={() => onSelect(entry.path)}
            >
              <bdi>{entry.path.split("/").at(-1)}</bdi>
              <span>{formatBytes(entry.size)}</span>
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

export function ArchivePreviewPanel({
  metadata,
  scope,
  loadOnMount = false,
}: {
  metadata: ArtifactMetadata;
  scope: ArtifactArchiveScope;
  loadOnMount?: boolean;
}) {
  const api = usePublicAPI();
  const [requested, setRequested] = useState(loadOnMount);
  const [selection, setSelection] = useState<string | null>(null);
  const key = archiveQueryKey(scope, metadata.artifact);
  const archive = useQuery({
    queryKey: key,
    queryFn: ({ signal }) =>
      getArtifactArchive(api, scope, metadata.artifact, signal),
    enabled: requested,
    retry: false,
    staleTime: Infinity,
    gcTime: 0,
  });
  const entries = archive.data?.entries;
  const selected =
    selection ??
    entries?.find(
      (entry) =>
        entry.kind === "file" &&
        entry.previewable &&
        (entry.path === "SKILL.md" || entry.path.endsWith("/SKILL.md")),
    )?.path ??
    null;
  const entry = entries?.find((item) => item.path === selected);
  const file = useQuery({
    queryKey: [...key, "file", selected],
    queryFn: ({ signal }) =>
      getArtifactArchiveFile(api, scope, metadata.artifact, selected!, signal),
    enabled: requested && selected !== null && entry?.previewable === true,
    retry: false,
    staleTime: Infinity,
    gcTime: 0,
  });
  const childrenByPath = useMemo(() => {
    const children = new Map<string, ArtifactArchiveEntry[]>();
    for (const item of entries ?? []) {
      const parent = item.path.slice(
        0,
        Math.max(0, item.path.lastIndexOf("/")),
      );
      const siblings = children.get(parent) ?? [];
      siblings.push(item);
      children.set(parent, siblings);
    }
    for (const siblings of children.values())
      siblings.sort((a, b) =>
        a.kind === b.kind
          ? a.path.localeCompare(b.path)
          : a.kind === "directory"
            ? -1
            : 1,
      );
    return children;
  }, [entries]);

  return (
    <div className="panel artifact-preview-panel archive-preview-panel">
      <div className="section-heading">
        <div>
          <h3>Files</h3>
        </div>
        <button
          className={entries === undefined ? undefined : "secondary-button"}
          type="button"
          disabled={archive.isFetching}
          onClick={() => {
            if (requested) void archive.refetch();
            else setRequested(true);
          }}
        >
          {archive.isFetching
            ? "Loading files…"
            : entries === undefined
              ? "Browse files"
              : "Reload files"}
        </button>
      </div>
      <p className="muted-copy">
        Choose a file to preview. Text files up to 256 KiB are supported.
      </p>
      {archive.error === null ? null : <ErrorNotice error={archive.error} />}
      {entries?.length === 0 ? (
        <p className="compact-empty">This archive is empty.</p>
      ) : null}
      {entries === undefined || entries.length === 0 ? null : (
        <div className="archive-browser">
          <nav aria-label="Archive files" className="archive-file-list">
            <ArchiveTree
              childrenByPath={childrenByPath}
              selected={selected}
              onSelect={setSelection}
            />
          </nav>
          <div className="archive-file-content" aria-live="polite">
            {entry === undefined ? (
              <p className="compact-empty">Select a file from the archive.</p>
            ) : (
              <>
                <div className="archive-file-heading">
                  <strong>
                    <bdi>{entry.path}</bdi>
                  </strong>
                  <span>{formatBytes(entry.size)}</span>
                </div>
                {!entry.previewable ? (
                  <p className="compact-empty">
                    This file exceeds the preview limits. Download the original
                    archive to view it.
                  </p>
                ) : file.isPending || file.isFetching ? (
                  <p className="loading-copy" role="status">
                    Loading file…
                  </p>
                ) : file.error !== null ? (
                  <>
                    <ErrorNotice
                      error={file.error}
                      onRetry={() => void file.refetch()}
                    />
                  </>
                ) : (
                  <LoadedArtifactPreview
                    key={`${metadata.artifact.revision}:${entry.path}`}
                    mediaType={
                      /\.md$/i.test(entry.path) ? "text/markdown" : "text/plain"
                    }
                    source={file.data.text}
                  />
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
