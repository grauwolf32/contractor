import { Icon, type IconName } from "../../app/icon";

/** Decorative stroke icons of the V3B decision bar and issue summary. */
type DecisionIconName = "confirm" | "reject" | "unsure" | "next";

const ICONS: Readonly<Record<DecisionIconName, IconName>> = {
  confirm: "check",
  reject: "x-circle",
  unsure: "question-circle",
  next: "arrow-down",
};

export function DecisionIcon({
  name,
  size = 16,
}: {
  name: DecisionIconName;
  size?: number;
}) {
  return <Icon name={ICONS[name]} size={size} />;
}
