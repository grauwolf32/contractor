import { useId } from "react";
import { useNavigate } from "react-router";
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
              <span className="start-scope-size">{summary.size}</span>
              <span className="ui-caption start-scope-detail">
                {`${presentCheckType(family.preferred).label} · ${summary.detail}`}
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
  return (
    <section className="start-card" aria-labelledby={`${id}-heading`}>
      <div className="start-card-head">
        <h3 id={`${id}-heading`}>Scope</h3>
      </div>
      <p className="start-scope-size">{summary.size}</p>
      <p className="ui-caption">{summary.detail}</p>
    </section>
  );
}
