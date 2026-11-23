import { Field } from "../../../ui";
import type { StartCheck } from "./use-start-check";

const descriptions: Record<
  string,
  { label: string; hint: string; placeholder?: string }
> = {
  server: {
    label: "API server URL",
    hint: "Exact HTTP or HTTPS base URL allowed by the engagement.",
    placeholder: "https://staging.example.com/api",
  },
  operations: {
    label: "Selected operations",
    hint: "JSON array of OpenAPI operation pointers, up to 16 operations.",
    placeholder: '["#/paths/~1pets/get"]',
  },
  test_parameters: {
    label: "SQLMap test parameters",
    hint: "JSON array of request parameter names to test.",
    placeholder: '["search"]',
  },
  credential_id: {
    label: "Credential ID",
    hint: "Optional credential referenced by an identity in the engagement. Leave empty for anonymous requests.",
  },
  template_ids: {
    label: "Rule IDs",
    hint: "Comma-separated HTTP rule IDs or patterns. Omit to use the profile selection.",
  },
  tags: {
    label: "Rule tags",
    hint: "Optional comma-separated HTTP rule tags.",
  },
  severity: {
    label: "Rule severity",
    hint: "Optional comma-separated info, low, medium, high, critical or unknown.",
  },
};
export function ExecutionParameters({ model }: { model: StartCheck }) {
  if (model.parameterFields.length === 0) return null;
  return (
    <section className="start-card">
      <div className="start-card-head">
        <h3>Execution parameters</h3>
      </div>
      <div className="start-fields">
        {model.parameterFields.map((field) => {
          const description = descriptions[field.name];
          return (
            <Field
              key={field.name}
              label={description?.label ?? field.name.replaceAll("_", " ")}
              hint={description?.hint ?? "Saved with this check."}
              required={field.required}
            >
              {(control) => (
                <input
                  {...control}
                  value={field.value}
                  placeholder={description?.placeholder}
                  maxLength={4096}
                  onChange={(event) =>
                    model.setParameter(field.name, event.target.value)
                  }
                />
              )}
            </Field>
          );
        })}
      </div>
      {model.slots.some((slot) => slot.name === "request_values") && (
        <p className="ui-caption">
          Request values can be supplied as an optional material keyed by
          operation pointer, with parameters and body fields. Authentication and
          HTTP limits come from the engagement.
        </p>
      )}
    </section>
  );
}
