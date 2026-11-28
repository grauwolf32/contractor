import type { ArtifactMetadata } from "../../api/artifacts";
import { ContextLink } from "../../app/context-navigation";
import { formatBytes } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";
import { Icon } from "../../app/icon";
import { IdChip, ListRow } from "../../ui";
import { MaterialIcon, MaterialKindMark } from "./icons";
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
  previewOnOpen = false,
}: {
  item: ArtifactMetadata;
  to: string;
  returnLabel: string;
  previewOnOpen?: boolean;
}) {
  const identity = item.artifact;
  const previewParams = new URLSearchParams(to.split("?")[1] ?? "");
  previewParams.set("revision", identity.revision);
  const previewTo = `${to.split("?")[0]}?${previewParams.toString()}`;
  return (
    <ListRow
      glyph={<MaterialKindMark kind={materialKindOf(item)} />}
      // Long names that differ only at the end must stay distinguishable.
      clamp={false}
      title={
        <ContextLink
          className="materials-row-link"
          to={previewOnOpen ? previewTo : to}
          aria-label={`${identity.namespace}/${identity.name}`}
          returnLabel={returnLabel}
        >
          {identity.name}
        </ContextLink>
      }
      meta={[
        <span key="format" title={item.mediaType}>
          {formatLabel(item.mediaType).includes("/")
            ? "File"
            : formatLabel(item.mediaType)}
        </span>,
        <span key="namespace" className="materials-ns">
          {identity.namespace}
        </span>,
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
      trailing={
        previewOnOpen ? (
          <Icon name="chevron-right" size={16} />
        ) : (
          <ContextLink
            className="materials-row-preview"
            to={previewTo}
            returnLabel={returnLabel}
            aria-label={`Preview ${identity.namespace}/${identity.name}`}
          >
            Preview →
          </ContextLink>
        )
      }
    />
  );
}
