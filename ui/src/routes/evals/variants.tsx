import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { getAuditProfile } from "../../api/audits";
import { getWorkflow } from "../../api/workflows";
import { type EvalCapabilities, type EvalVariant } from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { Field, IdChip } from "../../ui";
import {
  ArmKey,
  CommaSeparatedInput,
  EvalError,
  KeyValueEditor,
} from "./common";
import { executableLabel } from "./labels";
import { sortedBindings } from "./setup-model";

export function VariantEditor({
  value,
  onChange,
  capabilities,
  label,
}: {
  value: EvalVariant;
  onChange: (value: EvalVariant) => void;
  capabilities: EvalCapabilities | undefined;
  label: string;
}) {
  const api = usePublicAPI();
  const options = sortedBindings(capabilities, value.kind);
  const families = [...new Set(options.map((x) => x.selector.split("@")[0]!))];
  const [name = "", version = ""] = value.selector.split("@");
  const contract = useQuery({
    queryKey: queryKeys.evals.binding(value.kind, value.selector),
    enabled: !!value.selector,
    queryFn: async () =>
      value.kind === "workflow"
        ? getWorkflow(api, name, version)
        : getAuditProfile(api, name, version),
  });
  const inputs = contract.data?.inputs ?? {};
  const optionsForFamily = options.filter((x) =>
    x.selector.startsWith(name + "@"),
  );
  const arm = label === "A" ? "a" : "b";
  return (
    <fieldset className={`eval-variant eval-arm-${label.toLowerCase()}`}>
      <legend>
        <ArmKey arm={arm}>
          {label} · {label === "A" ? "Baseline" : "Candidate"}
        </ArmKey>
      </legend>
      <Field label={`${label} ${executableLabel(value.kind)} family`}>
        {(control) => (
          <select
            {...control}
            value={name}
            onChange={(e) => {
              const latest = options.find(
                (x) =>
                  x.available && x.selector.startsWith(e.target.value + "@"),
              );
              onChange({
                ...value,
                selector: latest?.selector ?? "",
              });
            }}
          >
            <option value="">Choose a family</option>
            {name && !families.includes(name) ? <option>{name}</option> : null}
            {families.map((family) => (
              <option
                key={family}
                disabled={
                  !options.some(
                    (x) => x.available && x.selector.startsWith(family + "@"),
                  )
                }
              >
                {family}
              </option>
            ))}
          </select>
        )}
      </Field>
      <Field
        label={`${label} version`}
        hint="A new family selection proposes its latest compatible version; Prepare pins the choice."
      >
        {(control) => (
          <select
            {...control}
            value={value.selector}
            onChange={(e) => onChange({ ...value, selector: e.target.value })}
          >
            <option value="">Choose a version</option>
            {value.selector &&
            !optionsForFamily.some((x) => x.selector === value.selector) ? (
              <option value={value.selector}>
                {value.selector} · retained, unavailable in catalog
              </option>
            ) : null}
            {optionsForFamily.map((x, index) => (
              <option
                value={x.selector}
                key={x.selector}
                disabled={!x.available}
              >
                {x.selector}
                {index === 0 ? " · latest in catalog" : ""}
                {x.reason ? ` · ${x.reason}` : ""}
              </option>
            ))}
          </select>
        )}
      </Field>
      <EvalError error={contract.error} />
      {contract.data ? (
        <div className="eval-contract">
          <p className="eval-contract-definition">
            <span>Definition</span>{" "}
            <IdChip
              value={`${contract.data.ref.name}@${contract.data.ref.version}`}
              display={`${contract.data.ref.name}@${contract.data.ref.version}`}
              label={`${label} definition`}
            />
          </p>
          {Object.entries(inputs).map(([key, slot]) => (
            <p key={key}>
              Input <code>{key}</code>:{" "}
              {slot.required ? "required" : "optional"} ·{" "}
              {slot.mediaTypes.join(", ")}
            </p>
          ))}
          {"parameters" in contract.data ? (
            <p>
              Parameters:{" "}
              {Object.entries(contract.data.parameters)
                .map(
                  ([key, slot]) =>
                    `${key}${slot.required ? " (required)" : ""}`,
                )
                .join(", ") || "none"}
            </p>
          ) : (
            <p>
              Check scope supports objective, target and authorizationScope.
            </p>
          )}
          {value.kind === "workflow" ? (
            <Link
              target="_blank"
              rel="noreferrer"
              to={`/catalog/workflows/${encodeURIComponent(name)}/${encodeURIComponent(version)}`}
            >
              Inspect Workflow, agents and Skills
            </Link>
          ) : null}
        </div>
      ) : null}
      <details className="eval-disclosure">
        <summary>Input/output mapping and parameters</summary>
        <div className="eval-disclosure-body">
          <p className="ui-caption">
            Map case roles to executable input/output names. Omitted roles
            retain their name. Use $task.objective for the visible task
            objective.
          </p>
          <KeyValueEditor
            label={`${label} input mapping`}
            value={value.inputMapping ?? {}}
            onChange={(inputMapping) => onChange({ ...value, inputMapping })}
          />
          <KeyValueEditor
            label={`${label} output mapping`}
            value={value.outputMapping ?? {}}
            onChange={(outputMapping) => onChange({ ...value, outputMapping })}
          />
          <KeyValueEditor
            label={`${label} parameters`}
            value={value.parameters ?? {}}
            onChange={(parameters) => onChange({ ...value, parameters })}
          />
        </div>
      </details>
      <details className="eval-disclosure">
        <summary>Execution settings</summary>
        <div className="eval-disclosure-body">
          <p className="ui-caption">
            {value.kind === "workflow"
              ? "Execution policies come from the selected Workflow."
              : "Execution policies come from the selected check type."}
          </p>
          <Field
            label={`${label} runtime labels`}
            hint="Optional labels, separated by commas; availability is checked during preparation."
          >
            {(control) => (
              <CommaSeparatedInput
                {...control}
                value={value.runtimeLabels ?? []}
                onChange={(runtimeLabels) =>
                  onChange({
                    ...value,
                    runtimeLabels,
                  })
                }
              />
            )}
          </Field>
        </div>
      </details>
    </fieldset>
  );
}
