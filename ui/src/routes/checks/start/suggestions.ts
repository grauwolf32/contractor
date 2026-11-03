/**
 * Static check type suggestions (docs/design/ui/v3b-implementation.md, "API
 * gaps": nothing on the Server suggests check types yet).
 *
 * A rule applies when the objective contains one of its words and one of
 * its check types is ready with the project's materials. Its reason is a
 * fixed sentence that says only what the rule checked: which words the
 * objective mentions and what the check type does. Materials are matched by
 * format alone, so the reason says "format matches" and never claims that a
 * material fits the objective (docs/spec/ui/journeys.md US-03).
 */

interface SuggestionRule {
  readonly id: string;
  /** Check type (profile) names this rule suggests, the better fit first. */
  readonly checkTypes: readonly string[];
  /** Words in the objective that make the rule apply. */
  readonly words: RegExp;
  /** Why this fits: a fixed sentence about the words and the check type. */
  readonly reason: string;
}

/** Second sentence of every reason: what readiness checked. */
const FORMAT_MATCH_REASON =
  "Your project has a material whose format matches each input it needs.";

/** In order: the first rule that applies is the suggestion. */
const SUGGESTION_RULES: readonly SuggestionRule[] = [
  {
    id: "asvs",
    checkTypes: [
      "owasp-asvs-5-0-l1-source-review",
      "owasp-asvs-5-0-l1-source-pilot",
    ],
    words: /\basvs\b/i,
    reason:
      "Your objective mentions ASVS, and this check type reviews the selected requirements against source and documentation.",
  },
  {
    id: "wstg",
    checkTypes: [
      "owasp-wstg-4-2-source-review",
      "owasp-wstg-4-2-fast-source-review",
    ],
    words: /\bwstg\b/i,
    reason:
      "Your objective mentions WSTG, and this check type reviews the selected scenarios against source and preserves live evidence gaps.",
  },
  {
    id: "top10",
    checkTypes: ["owasp-top10-2025-source-risk"],
    words: /\b(?:top[ -]?10|owasp top ten)\b/i,
    reason:
      "Your objective mentions Top 10, and this check type checks the selected source risk assertions.",
  },
  {
    id: "checklist",
    checkTypes: ["source-checklist"],
    words: /\b(?:checklist|assertions?)\b/i,
    reason:
      "Your objective mentions a checklist or assertions, and this check type checks their exact wording against selected source.",
  },
  {
    id: "sql-injection",
    checkTypes: ["openapi-sqlmap-scan"],
    words: /\b(?:sqlmap|sql[- ]?injections?|sqli)\b/i,
    reason:
      "Your objective mentions SQL injection, and this check type runs SQLMap on the endpoints your scan settings select.",
  },
  {
    id: "nuclei",
    checkTypes: ["openapi-nuclei-scan"],
    words: /\b(?:nuclei|misconfigurations?|security headers?)\b/i,
    reason:
      "Your objective mentions Nuclei, misconfiguration or security headers, and this check type runs Nuclei on the endpoints your scan settings select.",
  },
  {
    id: "api-access",
    checkTypes: ["openapi-operation-trace", "openapi-operation-observe"],
    words:
      /\b(?:apis?|endpoints?|routes?|rest|graphql|authori[sz]ation|access control|permissions?|idor|bola|object[- ]level)\b/i,
    reason:
      "Your objective mentions APIs, endpoints or access control, and this check type traces each endpoint of your API spec through the source code.",
  },
];

interface Suggestion {
  /** The suggested check type (profile name). */
  readonly checkType: string;
  readonly ruleId: string;
  /** The rule's reason followed by FORMAT_MATCH_REASON. */
  readonly reason: string;
}

export interface SuggestionResult {
  readonly suggestion?: Suggestion | undefined;
  /** A check type from a later rule, offered as "Or try …". */
  readonly alternative?: Suggestion | undefined;
}

/**
 * The suggested check type for an objective, and one alternative. `ready`
 * says whether a check type can start with the project's current
 * materials; rules never suggest one that cannot. A blank objective gets
 * no suggestion.
 */
export function suggestCheckTypes(
  objective: string,
  ready: (checkType: string) => boolean,
): SuggestionResult {
  const text = objective.trim();
  if (text === "") return {};
  const found: Suggestion[] = [];
  for (const rule of SUGGESTION_RULES) {
    if (!rule.words.test(text)) continue;
    const checkType = rule.checkTypes.find(
      (name) => ready(name) && !found.some((known) => known.checkType === name),
    );
    if (checkType === undefined) continue;
    found.push({
      checkType,
      ruleId: rule.id,
      reason: `${rule.reason} ${FORMAT_MATCH_REASON}`,
    });
    if (found.length === 2) break;
  }
  return { suggestion: found[0], alternative: found[1] };
}
