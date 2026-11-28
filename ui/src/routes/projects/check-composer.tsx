import { type FormEvent, useId, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { AuditProfile } from "../../api/audits";
import type { Project } from "../../api/projects";
import { Icon } from "../../app/icon";
import { artifactAccepts } from "../../run-drafts/validation";
import { Button, StatusGlyph } from "../../ui";
import { useAuditPresets } from "../catalog/audit-preset-data";
import { auditPresetLabel } from "../../modules/checks/labels";
import { useUnsavedDraft } from "./project-drafts";
import { startCheckPath } from "./project-sections";

/** Longest objective carried in the Start URL. */
const MAXIMUM_OBJECTIVE_LENGTH = 4_000;

function newerVersion(left: string, right: string): boolean {
  return left.localeCompare(right, "en", { numeric: true }) > 0;
}

/**
 * Check types whose required inputs the sampled materials match by media
 * type, judged on the newest version of each (the one `type=<name>` starts).
 * Only matches are claimed: the sample is bounded, so a check type left out
 * may still fit, and a match says nothing about the content. Active testing
 * also needs the project's live target.
 */
function formatMatches(
  profiles: readonly AuditProfile[],
  materials: readonly ArtifactMetadata[],
  project: Project,
): AuditProfile[] {
  const latest = new Map<string, AuditProfile>();
  for (const profile of profiles) {
    const known = latest.get(profile.ref.name);
    if (
      known === undefined ||
      newerVersion(profile.ref.version, known.ref.version)
    )
      latest.set(profile.ref.name, profile);
  }
  return [...latest.values()].filter((profile) => {
    const required = Object.values(profile.inputs).filter(
      (input) => input.required,
    );
    if (required.length === 0) return false;
    if (
      profile.policy.active_operations === "allowed" &&
      project.httpTarget === undefined
    )
      return false;
    return required.every((input) =>
      materials.some((material) =>
        artifactAccepts([...input.media_types], material),
      ),
    );
  });
}

function CheckTypeSuggestions({
  project,
  materials,
  objective,
}: {
  project: Project;
  materials: readonly ArtifactMetadata[];
  objective: string;
}) {
  const presets = useAuditPresets();
  const heading = useId();
  // Suggestions are a shortcut: while they load, or when the catalog cannot
  // be read, the composer and "Choose check type" still open setup.
  const suggestions =
    presets.data === undefined
      ? []
      : formatMatches(presets.data, materials, project).slice(0, 3);
  if (suggestions.length === 0) return null;
  return (
    <div className="projects-suggestions">
      <span className="projects-suggestions-heading" id={heading}>
        Check types whose input formats match your materials
      </span>
      <ul
        role="list"
        className="projects-suggestion-list"
        aria-labelledby={heading}
      >
        {suggestions.map((profile) => (
          <li key={profile.ref.name}>
            <Link
              className="projects-suggestion"
              to={startCheckPath(
                project.projectId,
                objective,
                profile.ref.name,
              )}
            >
              <span className="projects-suggestion-title">
                {auditPresetLabel(profile.ref.name)}
              </span>
              <span className="projects-suggestion-fit">
                <StatusGlyph tone="success" size={14} />
                Format matches
              </span>
              <span className="projects-suggestion-ref">
                {profile.ref.name}@{profile.ref.version}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * "What do you want to check?": the objective in plain words opens Start a
 * check for this project with the objective filled in.
 */
export function CheckComposer({
  project,
  materials,
}: {
  project: Project;
  materials: readonly ArtifactMetadata[];
}) {
  const navigate = useNavigate();
  const label = useId();
  const field = useId();
  const [objective, setObjective] = useState("");
  useUnsavedDraft(objective.trim() !== "");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void navigate(startCheckPath(project.projectId, objective));
  }

  return (
    <form
      className="projects-composer"
      aria-labelledby={label}
      onSubmit={submit}
    >
      <label className="projects-composer-title" id={label} htmlFor={field}>
        What do you want to check?
      </label>
      <div className="projects-composer-field">
        <textarea
          id={field}
          rows={2}
          maxLength={MAXIMUM_OBJECTIVE_LENGTH}
          placeholder="Describe it in plain words, for example: can one customer see another customer's orders?"
          aria-keyshortcuts="Control+Enter Meta+Enter"
          value={objective}
          onChange={(event) => setObjective(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <div className="projects-composer-footer">
          <Link to={startCheckPath(project.projectId, objective)}>
            <Icon name="catalog" size={14} /> Choose check type
          </Link>
          <Button type="submit" variant="primary" size="sm">
            Start a check <Icon name="arrow-right" size={15} />
          </Button>
        </div>
      </div>
      <CheckTypeSuggestions
        project={project}
        materials={materials}
        objective={objective}
      />
    </form>
  );
}
