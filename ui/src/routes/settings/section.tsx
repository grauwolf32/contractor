import "../operations/settings/settings.css";
import type { ReactNode } from "react";

import { TechnicalDetails } from "../../ui";
import { Icon, type IconName } from "../../app/icon";
import { ScopeChip } from "../operations/common";

const SECTION_ICONS: Readonly<Record<string, IconName>> = {
  "model-selection": "library",
  "workflow-scheduling": "runs",
  "repository-access": "binding",
  appearance: "settings",
};

const SECTION_INTROS: Readonly<Record<string, string>> = {
  "model-selection": "One default for new Runs and checks.",
  "workflow-scheduling": "Separate lanes for model and tool work.",
  "repository-access":
    "For private Git imports. Public HTTPS repositories do not need a key.",
  appearance: "Choose how Contractor looks in this browser.",
};

export type SettingHeadingLevel = "h2" | "h3";

/**
 * A Settings card with its scope, concise introduction, full explanation on
 * demand and the existing resource editor.
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
        <header className="ops-setting-card-head">
          <Icon name={SECTION_ICONS[id] ?? "settings"} size={18} />
          <div>
            <p className="ops-eyebrow">{eyebrow}</p>
            <Heading id={headingId} className="ops-setting-title">
              {title}
            </Heading>
          </div>
          <ScopeChip>{scope}</ScopeChip>
        </header>
        <div className="ops-setting-copy">
          <p>{SECTION_INTROS[id]}</p>
          <TechnicalDetails
            summary={`About ${typeof title === "string" ? title.toLowerCase() : "this setting"}`}
          >
            {about}
          </TechnicalDetails>
        </div>
      </div>
      <div className="ops-setting-editor">{children}</div>
    </section>
  );
}
