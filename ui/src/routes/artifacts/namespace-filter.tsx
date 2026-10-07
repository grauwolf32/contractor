import { type FormEvent, type ReactNode, useState } from "react";

import { ARTIFACT_NAME_PATTERN } from "../../api/artifacts";
import "./materials.css";

/**
 * Namespace filter form and its validation message. The form applies a
 * valid namespace, or undefined for all namespaces.
 */
export function useNamespaceFilter({
  value,
  onApply,
}: {
  value: string | undefined;
  onApply: (namespace: string | undefined) => void;
}): { form: ReactNode; error: ReactNode } {
  const [error, setError] = useState<string | null>(null);
  function apply(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const candidate = String(
      new FormData(event.currentTarget).get("namespaceFilter") ?? "",
    ).trim();
    if (candidate !== "" && !ARTIFACT_NAME_PATTERN.test(candidate)) {
      setError(
        "A namespace uses 1–128 letters, digits, dots, dashes or underscores and starts with a letter or digit.",
      );
      return;
    }
    setError(null);
    onApply(candidate === "" ? undefined : candidate);
  }
  return {
    form: (
      <form className="inline-form materials-filter" onSubmit={apply}>
        <label className="materials-filter-label">
          Namespace
          <input
            name="namespaceFilter"
            placeholder="all namespaces"
            autoComplete="off"
            spellCheck={false}
            key={value ?? ""}
            defaultValue={value ?? ""}
          />
        </label>
        <button className="ui-btn" data-size="sm" type="submit">
          Apply
        </button>
      </form>
    ),
    error:
      error === null ? null : (
        <p className="form-error" role="alert">
          {error}
        </p>
      ),
  };
}
