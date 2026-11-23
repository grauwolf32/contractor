import type { stringRule } from "../../../api/contract";

/** An exact published version: RuntimeConfig or LLM Gateway. */
interface ExactRef {
  name: string;
  version: string;
  digest: string;
}

/** The option value and React key of an exact version. */
export function exactKey(ref: ExactRef): string {
  return `${ref.name}@${ref.version}:${ref.digest}`;
}

/**
 * The sentence that rejects a value of `rule`, read from the rule itself so
 * that it names the grammar and length the Server enforces:
 * "Name must match [a-z][a-z0-9_-]* and contain at most 63 characters."
 */
export function ruleMessage(
  subject: string,
  rule: ReturnType<typeof stringRule>,
  ...more: string[]
): string {
  const pattern = rule.pattern?.source.replace(/^\^/, "").replace(/\$$/, "");
  const parts = [
    ...(pattern === undefined ? [] : [`match ${pattern}`]),
    ...(rule.maxLength === undefined
      ? []
      : [`contain at most ${rule.maxLength} characters`]),
    ...more,
  ];
  const last = parts.pop() ?? "be valid";
  return parts.length === 0
    ? `${subject} must ${last}.`
    : `${subject} must ${parts.join(", ")} and ${last}.`;
}
