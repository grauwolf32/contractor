import { useSharedConfig } from "../../../config/shared-config";
import { auditPresetPath } from "../../../api/audit-presets";
import { ContextLink } from "../../../app/context-navigation";
import { compactDigest } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { checkItemKind, itemCount } from "../../../app/vocabulary";
import {
  Facts,
  Field,
  LabelsEditor,
  TechnicalDetails,
  IdChip,
} from "../../../ui";
import { ScopeChoices } from "./scope";
import { TextAreaField, TextField } from "./fields";
import { SCOPE_TEXT_LIMITS } from "./request";
import type { StartCheck } from "./use-start-check";

/**
 * Advanced options, collapsed: the exact version and its task and attempt limits, the target and
 * authorization scope when the check type does not ask for them itself, and
 * runtime labels. The digest and input check sit behind Technical details.
 */
export function AdvancedOptions({ model }: { model: StartCheck }) {
  const { limits } = useSharedConfig();
  const { selection, exact } = model;
  if (selection === undefined) return null;
  const profile = exact ?? selection.profile;
  const { family } = selection;
  const selectable = family.versions.length > 1;
  const kind = checkItemKind(profile);
  const usesTarget =
    model.webScan !== undefined || model.scopeFields.has("target");
  const usesAuthorizationScope = model.scopeFields.has("authorizationScope");
  const invalidLabels = model.parsedLabels.invalid;
  return (
    <details className="start-advanced">
      <summary>
        <span className="start-advanced-title">
          <Icon name="chevron-right" size={13} strokeWidth={2} />
          Advanced options
        </span>
        <span className="ui-caption">Labels, version and check limits.</span>
      </summary>
      <div className="start-advanced-body">
        <ScopeChoices model={model} />
        <IdChip
          value={`${profile.ref.name}@${profile.ref.version}`}
          display={`${profile.ref.name}@${profile.ref.version}`}
          label="check type version"
        />
        <Facts
          items={[
            {
              term: "Check type version",
              value: selectable ? undefined : profile.ref.version,
            },
            {
              term: "Item limit",
              value: itemCount(kind, profile.limits.max_tasks),
            },
            {
              term: "Attempts per item",
              value: profile.limits.max_attempts,
            },
          ]}
        />
        <div className="start-fields">
          <LabelsEditor
            entries={model.labels}
            onChange={model.setLabels}
            maximum={limits.auditMaxLabels}
            suggestions={["cloud", "local"]}
            hint="cloud or local chooses where to run. Add a value for a provider or site, such as aws or lab. Labels are fixed at start and passed to every step."
            error={model.labelError}
          />
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
          <TextField
            label="Runtime configuration labels"
            value={model.runtimeLabels}
            onChange={model.setRuntimeLabels}
            placeholder="debug, caido"
            hint="Comma-separated configuration bindings."
            error={
              invalidLabels.length === 0
                ? undefined
                : `Not a runtime label: ${invalidLabels.join(", ")}. Labels start with a lowercase letter and use a–z, 0–9, - and _.`
            }
          />
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
                value: "Yes",
              },
            ]}
          />
        </TechnicalDetails>
      </div>
    </details>
  );
}
