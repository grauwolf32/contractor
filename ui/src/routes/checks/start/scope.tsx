import { ENGAGEMENT_MEDIA_TYPE } from "../../../api/artifacts";
import { useId } from "react";
import { useNavigate } from "react-router";
import { artifactDetailPath } from "../../artifacts/paths";
import { ButtonLink } from "../../../ui";
import { presentCheckType, scopeSummary } from "./check-types";
import type { StartCheck } from "./use-start-check";

export function ScopeChoices({ model }: { model: StartCheck }) {
  const id = useId();
  const navigate = useNavigate();
  const { selection } = model;
  if (selection === undefined || selection.row.row.variants.length <= 1)
    return null;
  return (
    <fieldset className="start-review-coverage">
      <legend>Review coverage</legend>
      {selection.row.row.variants.map((family) => {
        const summary = scopeSummary(family.preferred);
        return (
          <label key={family.name} className="start-scope-option">
            <input
              type="radio"
              name={`${id}-coverage`}
              checked={family.name === selection.family.name}
              onChange={() =>
                void navigate(model.href({ type: family.name }), {
                  replace: true,
                })
              }
            />
            <span>
              <span className="start-scope-size">
                {summary?.size ??
                  `Up to ${family.preferred.limits.max_tasks} tasks`}
              </span>
              <span className="ui-caption start-scope-detail">
                {presentCheckType(family.preferred).label}
              </span>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

export function ScopeCard({ model }: { model: StartCheck }) {
  const id = useId();
  const { selection } = model;
  if (selection === undefined) return null;
  const summary = scopeSummary(model.exact ?? selection.profile);
  if (summary === undefined) return null;
  const slots = model.slots.filter(
    (slot) =>
      slot.selected !== undefined &&
      (slot.name === "openapi" ||
        slot.input.media_types.includes(ENGAGEMENT_MEDIA_TYPE)),
  );
  return (
    <section className="start-card" aria-labelledby={`${id}-heading`}>
      <div className="start-card-head">
        <h3 id={`${id}-heading`}>Scope</h3>
      </div>
      <p className="start-scope-size">{summary.size}</p>
      <p className="ui-caption">{summary.detail}</p>
      {slots.map((slot) => (
        <div className="start-scope-material" key={slot.name}>
          <span>{slot.label}</span>
          <ButtonLink
            variant="ghost"
            size="sm"
            to={artifactDetailPath(
              { kind: "project", id: model.projectId },
              slot.selected!.artifact,
            )}
          >
            Preview material →
          </ButtonLink>
        </div>
      ))}
    </section>
  );
}
