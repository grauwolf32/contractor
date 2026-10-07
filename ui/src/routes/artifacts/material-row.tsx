import type { ArtifactMetadata } from "../../api/artifacts";
import { ContextLink } from "../../app/context-navigation";
import { formatBytes } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";
import { IdChip, ListRow } from "../../ui";
import { MaterialIcon, MaterialKindIcon } from "./icons";
import { formatLabel, materialKindOf } from "./kinds";
import "./materials.css";

/** The commit a Git import recorded (S24:26-27), short with a copy action. */
export function GitCommitChip({ commit }: { commit: string }) {
  return (
    <span className="materials-commit">
      <MaterialIcon name="git" size={13} />
      <span className="ui-visually-hidden">Git commit </span>
      <IdChip value={commit} label="Git commit" />
    </span>
  );
}

/** "namespace/name" with a quiet namespace; reads as one identifier. */
export function MaterialName({
  namespace,
  name,
}: {
  namespace: string;
  name: string;
}) {
  return (
    <>
      <span className="materials-ns">{namespace}/</span>
      {name}
    </>
  );
}

/**
 * One material or file in a list: its kind icon, `namespace/name` (a link
 * that keeps the list's return context), then format, size, when the
 * current version was added, the Git commit of an import and a lock.
 */
export function MaterialRow({
  item,
  to,
  returnLabel,
}: {
  item: ArtifactMetadata;
  to: string;
  returnLabel: string;
}) {
  const identity = item.artifact;
  return (
    <ListRow
      glyph={<MaterialKindIcon kind={materialKindOf(item)} />}
      title={
        <ContextLink
          className="materials-row-link"
          to={to}
          returnLabel={returnLabel}
        >
          <MaterialName namespace={identity.namespace} name={identity.name} />
        </ContextLink>
      }
      meta={[
        formatLabel(item.mediaType),
        formatBytes(item.size),
        <RecordedTime key="created" value={item.createdAt} />,
        item.gitSource === undefined ? null : (
          <GitCommitChip key="commit" commit={item.gitSource.resolvedCommit} />
        ),
        item.frozen ? (
          <span key="locked" className="materials-locked">
            <MaterialIcon name="lock" size={12} />
            Locked
          </span>
        ) : null,
      ]}
    />
  );
}
