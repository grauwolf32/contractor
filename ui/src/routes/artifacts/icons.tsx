import { Icon } from "../../app/icon";
import type { MaterialKind } from "./kinds";
import "./material-library.css";

export type MaterialIconName =
  | MaterialKind
  | "git"
  | "plus"
  | "upload"
  | "download"
  | "history"
  | "lock"
  | "close";

/** Decorative stroke icon; the text next to it carries the meaning. */
export function MaterialIcon({
  name,
  size = 16,
}: {
  name: MaterialIconName;
  size?: number;
}) {
  return (
    <Icon
      name={name === "engagement" ? "lock" : name === "other" ? "docs" : name}
      size={size}
    />
  );
}

/** The icon of a material kind, as on the project's material chips. */
function MaterialKindIcon({
  kind,
  size,
}: {
  kind: MaterialKind;
  size?: number;
}) {
  return <MaterialIcon name={kind} size={size ?? 16} />;
}

/** Shared coloured identity in Materials and the New check chooser. */
export function MaterialKindMark({ kind }: { kind: MaterialKind }) {
  return (
    <span className="materials-kind-mark" data-kind={kind}>
      <MaterialKindIcon kind={kind} size={18} />
    </span>
  );
}
