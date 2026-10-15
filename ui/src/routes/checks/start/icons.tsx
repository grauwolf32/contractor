// Small stroke icons of the Start page (docs/design/ui/mockups/v3b/start.html).
// Decorative: every one is aria-hidden next to its words.
import { Icon } from "../../../app/icon";

export function CheckIcon({ size = 14 }: { size?: number }) {
  return <Icon name="check" size={size} strokeWidth={2.2} />;
}

export function BulbIcon({ size = 14 }: { size?: number }) {
  return <Icon name="bulb" size={size} strokeWidth={2} />;
}

export function PlayIcon({ size = 16 }: { size?: number }) {
  return <Icon name="play" size={size} />;
}

export function ChevronIcon({ size = 13 }: { size?: number }) {
  return <Icon name="chevron-right" size={size} strokeWidth={2} />;
}
