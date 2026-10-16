import { auditPresetPath } from "../../../api/audit-presets";
import { ContextLink } from "../../../app/context-navigation";
import { compactDigest } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { checkItemKind, itemCount } from "../../../app/vocabulary";
import { Facts, Field, TechnicalDetails } from "../../../ui";
import { MODE_LABELS } from "./check-types";
import { TextAreaField, TextField } from "./fields";
import { SCOPE_TEXT_LIMITS } from "./request";
import type { StartCheck } from "./use-start-check";

/**
 * Advanced options, collapsed: the exact version and what it pins (mode,
 * item limit, attempts, standards and their selection), the target and
 * authorization scope when the check type does not ask for them itself, and
 * runtime labels. The digest and input check sit behind Technical details.
 */
export function AdvancedOptions({ model }: { model: StartCheck }) {
  const { selection, exact } = model;
  if (selection === undefined) return null;
  const profile = exact ?? selection.profile;
  const { family } = selection;
  const selectable = family.versions.length > 1;
  const kind = checkItemKind(profile);
  const standardSelection = profile.inventory.standardSelection;
  const usesTarget = model.scopeFields.has("target");
  const usesAuthorizationScope = model.scopeFields.has("authorizationScope");
  const invalidLabels = model.parsedLabels.invalid;
  return (
    <details className="start-advanced">
      <summary>
        <span className="start-advanced-title">
          <Icon name="chevron-right" size={13} strokeWidth={2} />
          Advanced options
        </span>
        <span className="ui-caption">
          Version, limits, standards, runtime labels and more. The defaults suit
          most checks.
        </span>
      </summary>
      <div className="start-advanced-body">
        <Facts
          items={[
            {
              term: "Check type version",
              value: selectable ? undefined : profile.ref.version,
            },
            {
              term: "Mode",
              value: MODE_LABELS[profile.mode] ?? profile.mode,
            },
            {
              term: "Item limit",
              value: itemCount(kind, profile.execution.maxItemsTotal),
            },
            {
              term: "Attempts per item",
              value: profile.execution.maxItemRunAttempts,
            },
          ]}
        />
        {profile.standards.length === 0 ? null : (
          <p className="start-tight" data-testid="start-standards">
            Standards pinned at start:{" "}
            {profile.standards
              .map((standard) => `${standard.scheme}@${standard.version}`)
              .join(", ")}
          </p>
        )}
        {standardSelection === undefined ? null : (
          <div
            className="start-selection"
            data-testid="start-standard-selection"
          >
            <strong>{standardSelection.scope}</strong>
            <p className="start-tight">
              Levels {standardSelection.levels.join(", ")} ·{" "}
              {itemCount(kind, standardSelection.entryIds.length)}
            </p>
          </div>
        )}
        <div className="start-fields">
          {selectable ? (
            <Field label="Check type version">
              {(control) => (
                <select
                  {...control}
                  value={selection.profile.ref.version}
                  onChange={(event) => model.chooseVersion(event.target.value)}
                >
                  {family.versions.map((version, index) => (
                    <option
                      key={version.ref.version}
                      value={version.ref.version}
                    >
                      {version.ref.version}
                      {index === 0 ? " (newest)" : ""}
                      {version.serverCompatible
                        ? ""
                        : " · not supported by this server"}
                    </option>
                  ))}
                </select>
              )}
            </Field>
          ) : null}
          {usesTarget ? null : (
            <TextField
              label="Target"
              value={model.target}
              onChange={model.setTarget}
              maxLength={SCOPE_TEXT_LIMITS.target}
              hint="Optional. Saved with the check."
            />
          )}
          {usesAuthorizationScope ? null : (
            <TextAreaField
              label="Authorization scope"
              value={model.authorizationScope}
              onChange={model.setAuthorizationScope}
              maxLength={SCOPE_TEXT_LIMITS.authorizationScope}
              hint="Optional. What you are allowed to test, saved with the check."
            />
          )}
          <TextField
            label="Runtime labels"
            value={model.runtimeLabels}
            onChange={model.setRuntimeLabels}
            placeholder="debug, caido"
            hint="Separate labels with commas or spaces. They choose the runtime configuration."
            error={
              invalidLabels.length === 0
                ? undefined
                : `Not a runtime label: ${invalidLabels.join(", ")}. Labels start with a lowercase letter and use a–z, 0–9, - and _.`
            }
          />
        </div>
        <p className="start-tight">
          <ContextLink
            to={auditPresetPath(profile.ref.name, profile.ref.version)}
            returnLabel="Start a check"
          >
            See what this check type covers in the Library
          </ContextLink>
        </p>
        <TechnicalDetails description="For admins and debugging.">
          <Facts
            size="sm"
            items={[
              {
                term: "Digest",
                value: (
                  <code title={profile.ref.digest}>
                    {compactDigest(profile.ref.digest)}
                  </code>
                ),
              },
              {
                term: "Inputs validated at start",
                value: profile.requiresInputValidation ? "Yes" : "No",
              },
            ]}
          />
        </TechnicalDetails>
      </div>
    </details>
  );
}
