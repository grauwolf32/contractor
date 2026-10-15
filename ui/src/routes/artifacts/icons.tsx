import { Icon } from "../../app/icon";
import type { MaterialKind } from "./kinds";

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
  return <Icon name={name} size={size} className="materials-icon" />;
}

/** The icon of a material kind, as on the project's material chips. */
export function MaterialKindIcon({
  kind,
  size,
}: {
  kind: MaterialKind;
  size?: number;
}) {
  return <MaterialIcon name={kind} size={size ?? 16} />;
}
