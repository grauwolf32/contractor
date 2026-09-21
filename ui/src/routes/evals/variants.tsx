import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { getAuditProfile } from "../../api/audits";
import {
  getWorkflow,
  listConfigurations,
  listCredentials,
} from "../../api/workflows";
import {
  evalInventory,
  type EvalCapabilities,
  type EvalVariant,
} from "../../api/evals";
import {
  CommaSeparatedInput,
  EvalError,
  EvalField,
  KeyValueEditor,
} from "./common";
import { sortedBindings } from "./setup-model";

function ExecutionOverrides({
  value,
  onChange,
}: {
  value: EvalVariant["executionConfig"];
  onChange: (value: EvalVariant["executionConfig"]) => void;
}) {
  const api = usePublicAPI();
  const inventory = useQuery({
    queryKey: ["evals", "execution-choices"],
    queryFn: async ({ signal }) => {
      const [models, gateways, credentials] = await Promise.all([
        evalInventory(
          (cursor) =>
            listConfigurations(api, "model-policies", {
              signal,
              ...(cursor ? { cursor } : {}),
            }),
          signal,
        ),
        evalInventory(
          (cursor) =>
            listConfigurations(api, "llm-gateways", {
              signal,
              ...(cursor ? { cursor } : {}),
            }),
          signal,
        ),
        evalInventory(
          (cursor) => listCredentials(api, cursor ? { cursor } : {}),
          signal,
        ),
      ]);
      return {
        models: models.map((x) => `${x.ref.name}@${x.ref.version}`),
        gateways: gateways.map((x) => `${x.ref.name}@${x.ref.version}`),
        credentials: credentials.map((x) => x.credentialId),
      };
    },
  });
  return (
    <>
      <EvalError error={inventory.error} />
      {(["planner", "workers"] as const).map((role) => (
        <fieldset key={role}>
          <legend>{role === "planner" ? "Planner" : "Workers"}</legend>
          {(
            [
              {
                field: "modelPolicy",
                title: "Model policy",
                choices: inventory.data?.models,
              },
              {
                field: "llmGateway",
                title: "Gateway",
                choices: inventory.data?.gateways,
              },
              {
                field: "credential",
                title: "Credential",
                choices: inventory.data?.credentials,
              },
            ] as const
          ).map(({ field, title, choices }) => {
            const selected = value[role]?.[field];
            const display = selected === null ? "__none__" : (selected ?? "");
            return (
              <EvalField label={`${role} ${title}`} key={field}>
                <select
                  value={display}
                  onChange={(e) => {
                    const patch = { ...value[role] };
                    if (!e.target.value) delete patch[field];
                    else
                      patch[field] =
                        e.target.value === "__none__"
                          ? (null as never)
                          : e.target.value;
                    onChange({ ...value, [role]: patch });
                  }}
                >
                  <option value="">Use selected Workflow default</option>
                  {field === "credential" ? (
                    <option value="__none__">No credential</option>
                  ) : null}
                  {display &&
                  display !== "__none__" &&
                  !choices?.includes(display) ? (
                    <option value={display}>
                      {display} · retained selection
                    </option>
                  ) : null}
                  {choices?.map((option) => (
                    <option key={option}>{option}</option>
                  ))}
                </select>
              </EvalField>
            );
          })}
        </fieldset>
      ))}
      {value.stages && Object.keys(value.stages).length ? (
        <p>
          Per-stage overrides from the saved draft are retained. Review them in
          Setup before Prepare.
        </p>
      ) : null}
    </>
  );
}

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
    queryKey: ["evals", "binding", value.kind, value.selector],
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
  return (
    <fieldset className={`eval-variant eval-arm-${label.toLowerCase()}`}>
      <legend>
        {label} · {label === "A" ? "Baseline" : "Candidate"}
      </legend>
      <EvalField
        label={`${label} ${value.kind === "audit" ? "AuditProfile" : "Workflow"} family`}
      >
        <select
          value={name}
          onChange={(e) => {
            const latest = options.find(
              (x) => x.available && x.selector.startsWith(e.target.value + "@"),
            );
            onChange({
              ...value,
              selector: latest?.selector ?? "",
              executionConfig: {},
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
      </EvalField>
      <EvalField
        label={`${label} version`}
        hint="A new family selection proposes its latest compatible version; Prepare pins the choice."
      >
        <select
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
            <option value={x.selector} key={x.selector} disabled={!x.available}>
              {x.selector}
              {index === 0 ? " · latest in catalog" : ""}
              {x.reason ? ` · ${x.reason}` : ""}
            </option>
          ))}
        </select>
      </EvalField>
      <EvalError error={contract.error} />
      {contract.data ? (
        <div className="eval-contract">
          <p>
            Definition: {contract.data.ref.name}@{contract.data.ref.version}
          </p>
          {Object.entries(inputs).map(([key, slot]) => (
            <p key={key}>
              Input {key}: {slot.required ? "required" : "optional"} ·{" "}
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
              Audit scope supports objective, target and authorizationScope.
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
      <details>
        <summary>Input/output mapping and parameters</summary>
        <p>
          Map case roles to executable input/output names. Omitted roles retain
          their name. Use $task.objective for the visible task objective.
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
      </details>
      <details>
        <summary>Execution settings</summary>
        {value.kind === "workflow" ? (
          <ExecutionOverrides
            value={value.executionConfig}
            onChange={(executionConfig) =>
              onChange({ ...value, executionConfig })
            }
          />
        ) : (
          <p>Execution settings come from the pinned AuditProfile.</p>
        )}
        <EvalField
          label={`${label} runtime labels`}
          hint="Optional labels, separated by commas; availability is checked during preparation."
        >
          <CommaSeparatedInput
            value={value.runtimeLabels ?? []}
            onChange={(runtimeLabels) =>
              onChange({
                ...value,
                runtimeLabels,
              })
            }
          />
        </EvalField>
      </details>
    </fieldset>
  );
}
