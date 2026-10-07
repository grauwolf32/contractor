import "../operations/settings/settings.css";
import type { ReactNode } from "react";

import { ScopeChip } from "../operations/common";

export type SettingHeadingLevel = "h2" | "h3";

/**
 * One block of the Settings page: what the setting is and who it affects on
 * the left, the editor on the right (stacked on narrow screens).
 */
export function SettingSection({
  id,
  headingId = `${id}-heading`,
  titleAs: Heading = "h3",
  eyebrow,
  title,
  scope,
  about,
  children,
}: {
  /** Anchor of the section (the page directory links to it). */
  id: string;
  headingId?: string | undefined;
  titleAs?: SettingHeadingLevel | undefined;
  eyebrow: ReactNode;
  title: ReactNode;
  /** Who the setting applies to: "Server-wide", "This browser", … */
  scope: ReactNode;
  about: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="ops-setting" aria-labelledby={headingId}>
      <div className="ops-setting-about">
        <p className="ops-eyebrow">{eyebrow}</p>
        <Heading id={headingId} className="ops-setting-title">
          {title}
        </Heading>
        <ScopeChip>{scope}</ScopeChip>
        <div className="ops-setting-copy">{about}</div>
      </div>
      <div className="ops-setting-editor">{children}</div>
    </section>
  );
}
