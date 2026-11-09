import { useId } from "react";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { schemaLimit } from "../../../api/contract";
import { Button, Field, TechnicalDetails } from "../../../ui";
import type { ProjectMaterials } from "./data";
import type { WebScanDraft } from "./web-scan-state";

export function WebScanSetup({
  model,
  materials,
  busy,
}: {
  model: WebScanDraft;
  materials: ProjectMaterials;
  busy: boolean;
}) {
  const id = useId();
  const packs = (materials.items ?? []).filter((item) =>
    [
      "application/yaml",
      "application/vnd.contractor.nuclei-rulepack+zip",
    ].includes(item.mediaType),
  );
  const key = (ref: { namespace: string; name: string; revision: string }) =>
    `${ref.namespace}/${ref.name}@${ref.revision}`;
  const selectedPack = model.pack;
  if (
    selectedPack !== undefined &&
    !packs.some((pack) => key(pack.artifact) === key(selectedPack.artifact))
  )
    packs.push(selectedPack);
  return (
    <fieldset className="start-web-fields" disabled={busy}>
      <section className="start-card">
        <div className="start-card-head">
          <h3>Scan this project's website</h3>
        </div>
        <p className="ui-caption start-tight">
          Nuclei runs HTTP rules against one HTTP or HTTPS target. Network
          access, credentials and request limits come from the selected
          engagement. Progress, evidence and results are saved in this project's
          check history.
        </p>
        <Field
          label="Target URL"
          required
          hint={
            model.project?.httpTarget === undefined
              ? "Enter the address to check."
              : "Filled from the project. Review the address before starting."
          }
        >
          {(control) => (
            <input
              {...control}
              type="url"
              value={model.target}
              onChange={(event) => model.setTarget(event.target.value)}
              placeholder="https://staging.example.com"
              maxLength={schemaLimit(
                "AuditWebTarget",
                "maxLength",
                "properties",
                "url",
              )}
            />
          )}
        </Field>
      </section>
      <section className="start-card">
        <div className="start-card-head">
          <h3>Rules to run</h3>
        </div>
        <div
          className="start-rule-options"
          role="radiogroup"
          aria-label="Rule source"
        >
          {[
            {
              custom: false,
              title: "Installed rules",
              text: "Use the installed Nuclei package. Narrow the selection by rule ID, tags or severity.",
            },
            {
              custom: true,
              title: "Custom Nuclei rules",
              text: "Check this project with your own YAML rule or ZIP rulepack.",
            },
          ].map((option) => (
            <label
              key={option.title}
              className="start-rule-option"
              data-selected={model.custom === option.custom}
            >
              <input
                type="radio"
                name={`${id}-rules`}
                checked={model.custom === option.custom}
                onChange={() => model.setCustom(option.custom)}
              />
              <span>
                <strong>{option.title}</strong>
                <span className="ui-caption">{option.text}</span>
              </span>
            </label>
          ))}
        </div>
        {model.custom ? (
          <div className="ui-stack">
            <Field
              label="Project rulepack"
              hint="Choose an exact revision already saved in this project, or upload a rule file below."
            >
              {(control) => (
                <select
                  {...control}
                  value={
                    model.file !== undefined || model.pack === undefined
                      ? ""
                      : key(model.pack.artifact)
                  }
                  onChange={(event) => {
                    model.setFile(undefined);
                    model.setPack(
                      packs.find(
                        (pack) => key(pack.artifact) === event.target.value,
                      ),
                    );
                  }}
                >
                  <option value="">Choose project rules</option>
                  {packs.map((pack) => (
                    <option key={key(pack.artifact)} value={key(pack.artifact)}>
                      {pack.artifact.namespace}/{pack.artifact.name} ·{" "}
                      {pack.artifact.revision}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            {!materials.complete ? (
              <Button
                size="sm"
                onClick={() => void materials.loadMore()}
                disabled={materials.loading || materials.loadingMore}
              >
                Load more project rules
              </Button>
            ) : null}
            {materials.error === null ? null : (
              <ErrorNotice error={materials.error} onRetry={materials.retry} />
            )}
            <Field
              label="Upload Nuclei rules"
              hint="YAML for one rule (up to 1 MiB), or ZIP for a rulepack (up to 16 MiB). Saved in this project when you save or start the check."
            >
              {(control) => (
                <input
                  {...control}
                  type="file"
                  accept=".yaml,.yml,.zip"
                  onChange={(event) => {
                    model.setFile(event.target.files?.[0]);
                    model.setPack(undefined);
                  }}
                />
              )}
            </Field>
            {model.file === undefined ? null : (
              <p className="ui-caption">
                Selected: {model.file.name} · {formatBytes(model.file.size)}
              </p>
            )}
          </div>
        ) : null}
      </section>
      <details className="start-advanced">
        <summary>
          Scan options <span className="ui-caption">Rule filters</span>
        </summary>
        <div className="start-advanced-body start-fields">
          <Field
            label="Rule IDs"
            hint="Optional comma-separated Nuclei IDs or patterns. Empty uses all eligible rules in your selected package."
          >
            {(control) => (
              <input
                {...control}
                value={model.templateIds}
                maxLength={schemaLimit("NucleiRuleFilter", "maxLength")}
                onChange={(event) => model.setTemplateIds(event.target.value)}
              />
            )}
          </Field>
          <Field label="Tags" hint="Optional comma-separated tags.">
            {(control) => (
              <input
                {...control}
                value={model.tags}
                maxLength={schemaLimit("NucleiRuleFilter", "maxLength")}
                onChange={(event) => model.setTags(event.target.value)}
              />
            )}
          </Field>
          <Field
            label="Severity"
            hint="Optional: info, low, medium, high, critical or unknown (comma-separated)."
          >
            {(control) => (
              <input
                {...control}
                value={model.severity}
                onChange={(event) => model.setSeverity(event.target.value)}
              />
            )}
          </Field>
        </div>
      </details>
      <p className="ui-caption">
        Only controlled HTTP templates are supported. Raw requests, JavaScript,
        flow, out-of-band callbacks and other unsupported constructs fail
        explicitly before scanning. The scan sends live requests and follows
        this project's active-check policy. Its target and rule inputs are fixed
        at start. Results record what ran and any gaps; scanner observations
        still need review.
      </p>
      <TechnicalDetails>
        <p>Web scan · web-scan@1 · Nuclei · no model required</p>
      </TechnicalDetails>
    </fieldset>
  );
}
