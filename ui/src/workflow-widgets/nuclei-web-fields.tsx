import { Field } from "../ui";
import { nucleiWebWidget } from "./nuclei-web";

/** Nuclei controls, reused by ordinary Run and durable sweep hosts. */
export function NucleiWebFields({
  parameters,
  onChange,
  includeTarget,
}: {
  parameters: Readonly<Record<string, string>>;
  onChange: (name: string, value: string) => void;
  includeTarget: boolean;
}) {
  return (
    <div className="ui-stack">
      {nucleiWebWidget.fields
        .filter((field) => includeTarget || field.name !== "target")
        .map((field) => (
          <Field
            key={field.name}
            label={field.label}
            required={field.required ?? false}
            hint={field.hint}
          >
            {(control) => (
              <input
                {...control}
                type={field.type ?? "text"}
                inputMode={field.inputMode as "numeric" | undefined}
                placeholder={field.placeholder}
                value={parameters[field.name] ?? ""}
                onChange={(event) => onChange(field.name, event.target.value)}
              />
            )}
          </Field>
        ))}
    </div>
  );
}
