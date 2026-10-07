/**
 * Check types (AuditProfiles) as the Start page shows them: families of
 * versions, rows that fold scope variants together, plain-language labels,
 * and readiness against the project's current materials.
 *
 * Everything here is derived from the profile contract the Server publishes
 * (inputs, inventory, standards, interaction, workflows) and from the
 * project's current materials. Media types decide compatibility: a match
 * means the format matches, never that a material is the right one
 * (docs/spec/ui-user-stories.md US-03).
 */
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { AuditProfile } from "../../../api/audits";
import { checkItemKind, itemCount, itemNoun } from "../../../app/vocabulary";

export type ProfileInput = AuditProfile["inputs"][string];
export type CompatibilityReason = AuditProfile["compatibilityReasons"][number];

/**
 * Version order, oldest first: dotted numbers compare part by part
 * ("10" after "9", "1.10" after "1.9"); other versions compare naturally.
 */
export function compareVersions(left: string, right: string): number {
  const numeric = /^\d+(?:\.\d+)*$/;
  if (numeric.test(left) && numeric.test(right)) {
    const a = left.split(".").map(BigInt);
    const b = right.split(".").map(BigInt);
    for (let index = 0; index < Math.max(a.length, b.length); index++) {
      const x = a[index] ?? 0n;
      const y = b[index] ?? 0n;
      if (x !== y) return x > y ? 1 : -1;
    }
  }
  return (
    left.localeCompare(right, "en", { numeric: true }) ||
    (left < right ? -1 : left > right ? 1 : 0)
  );
}

/** All loaded versions of one check type name. */
export interface CheckTypeFamily {
  readonly name: string;
  /** Newest first. */
  readonly versions: readonly AuditProfile[];
  /** The newest version this Server can run, else the newest version. */
  readonly preferred: AuditProfile;
}

export function groupFamilies(
  profiles: readonly AuditProfile[],
): CheckTypeFamily[] {
  const byName = new Map<string, AuditProfile[]>();
  for (const profile of profiles) {
    const versions = byName.get(profile.ref.name) ?? [];
    if (!versions.some((known) => known.ref.version === profile.ref.version))
      versions.push(profile);
    byName.set(profile.ref.name, versions);
  }
  return [...byName.entries()].map(([name, versions]) => {
    const sorted = [...versions].sort((a, b) =>
      compareVersions(b.ref.version, a.ref.version),
    );
    const preferred =
      sorted.find((profile) => profile.serverCompatible) ?? sorted[0]!;
    return { name, versions: sorted, preferred };
  });
}

/**
 * One row of the check type list. Check types that differ only in how much
 * they cover (the same standard, inputs and rules, a different selection or
 * item limit, such as a fast first pass) share a row; the Scope choice picks
 * between them.
 */
export interface CheckTypeRow {
  /** The variant that covers the most; it names the row. */
  readonly primary: CheckTypeFamily;
  /** Every variant, the one that covers the most first. */
  readonly variants: readonly CheckTypeFamily[];
}

/** How many items a check type works on at most. */
export function coverageSize(profile: AuditProfile): number {
  return (
    profile.inventory.standardSelection?.entryIds.length ??
    profile.execution.maxItemsTotal
  );
}

function inputContract(profile: AuditProfile): unknown[] {
  return Object.entries(profile.inputs)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, input]) => [
      name,
      input.required,
      [...input.mediaTypes].sort(),
    ]);
}

/** The key that makes two check types scope variants of each other. */
function variantKey(profile: AuditProfile): string | undefined {
  // Only standard-based checks: their variants select from one standard.
  if (profile.standards.length === 0) return undefined;
  const { interaction } = profile;
  return JSON.stringify([
    profile.mode,
    profile.inventory.implementation,
    [...new Set(profile.standards.map((standard) => standard.scheme))].sort(),
    inputContract(profile),
    [
      interaction.activeChecks,
      interaction.findingConfirmation,
      interaction.notApplicable,
      interaction.reportAcceptance,
    ],
  ]);
}

function byCoverage(left: CheckTypeFamily, right: CheckTypeFamily): number {
  return (
    coverageSize(right.preferred) - coverageSize(left.preferred) ||
    left.name.localeCompare(right.name)
  );
}

/** Rows in display order: known check types first, then by label. */
export function groupRows(
  families: readonly CheckTypeFamily[],
): CheckTypeRow[] {
  const keyed = new Map<string, CheckTypeFamily[]>();
  const rows: CheckTypeFamily[][] = [];
  for (const family of families) {
    const key = variantKey(family.preferred);
    const known = key === undefined ? undefined : keyed.get(key);
    if (known !== undefined) {
      known.push(family);
      continue;
    }
    const row = [family];
    if (key !== undefined) keyed.set(key, row);
    rows.push(row);
  }
  return rows
    .map((variants) => {
      const sorted = [...variants].sort(byCoverage);
      return { primary: sorted[0]!, variants: sorted };
    })
    .sort((left, right) => {
      const a = presentCheckType(left.primary.preferred);
      const b = presentCheckType(right.primary.preferred);
      return a.order - b.order || a.label.localeCompare(b.label);
    });
}

// ---------- Plain-language presentation ----------

export interface CheckTypePresentation {
  readonly label: string;
  /** One sentence: what the check type does. */
  readonly description: string;
  /** Display order; check types this UI does not know come last. */
  readonly order: number;
}

/**
 * The bundled catalog (configs/audit-profiles, docs/guides/audits.md). Other
 * check types get a label from their name and a description from their
 * contract, never a guess about their purpose.
 */
const KNOWN: Readonly<Record<string, CheckTypePresentation>> = {
  "openapi-operation-trace": {
    label: "API endpoint trace",
    description:
      "Traces every endpoint of your API spec through the source code and proposes possible issues.",
    order: 10,
  },
  "openapi-operation-observe": {
    label: "API endpoint map",
    description: "The same trace, as a map only. Proposes no issues.",
    order: 20,
  },
  "owasp-top10-2025-source-risk": {
    label: "OWASP Top 10 (2025) review",
    description:
      "Ten source code risk scenarios mapped to the OWASP Top 10:2025. A risk overview, not exhaustive.",
    order: 30,
  },
  "owasp-asvs-5-0-l1-source-review": {
    label: "OWASP ASVS 5.0 Level 1",
    description:
      "Verifies the source code and documentation against the ASVS 5.0 Level 1 requirements.",
    order: 40,
  },
  "owasp-asvs-5-0-l1-source-pilot": {
    label: "OWASP ASVS 5.0 Level 1 pilot",
    description: "Five selected ASVS 5.0 Level 1 requirements.",
    order: 41,
  },
  "owasp-wstg-4-2-source-review": {
    label: "OWASP WSTG 4.2 code review",
    description:
      "Reviews the source code against the WSTG 4.2 test scenarios. Sends no live traffic.",
    order: 50,
  },
  "owasp-wstg-4-2-fast-source-review": {
    label: "OWASP WSTG 4.2 fast code review",
    description:
      "A first pass over 16 priority WSTG scenarios; the others stay out of scope.",
    order: 51,
  },
  "owasp-wstg-4-2-active-http": {
    label: "OWASP WSTG 4.2 live testing",
    description:
      "Tests a running website or API over HTTP, scenario by scenario, with your approval for live tests.",
    order: 60,
  },
  "owasp-wstg-4-2-fast-active-http": {
    label: "OWASP WSTG 4.2 fast live testing",
    description:
      "A first pass over 16 priority WSTG scenarios against the running application.",
    order: 61,
  },
  "openapi-nuclei-scan": {
    label: "Nuclei scan",
    description:
      "Runs Nuclei on the endpoints your scan settings select, each after your approval.",
    order: 70,
  },
  "openapi-sqlmap-scan": {
    label: "SQLMap scan",
    description:
      "Runs SQLMap on the endpoints your scan settings select, each after your approval.",
    order: 80,
  },
  "source-checklist": {
    label: "Custom checklist",
    description: "Works through your own checklist against the source code.",
    order: 90,
  },
};

export const MODE_LABELS: Readonly<Record<AuditProfile["mode"], string>> = {
  "risk-assessment": "Risk assessment",
  "requirements-verification": "Requirements verification",
  "custom-checklist": "Custom checklist",
  "operation-tracing": "Operation tracing",
  "finding-verification": "Finding verification",
};

function humanize(name: string): string {
  const words = name.replaceAll(/[-_.]+/g, " ").trim();
  return words === "" ? name : words.charAt(0).toUpperCase() + words.slice(1);
}

/** "owasp-asvs 5.0.0, owasp-wstg 4.2" */
function standardsText(profile: AuditProfile): string {
  return profile.standards
    .map((standard) => `${standard.scheme} ${standard.version}`)
    .join(", ");
}

function contractDescription(profile: AuditProfile): string {
  const mode = Object.hasOwn(MODE_LABELS, profile.mode)
    ? MODE_LABELS[profile.mode]
    : humanize(profile.mode);
  const kind = checkItemKind(profile);
  const unit = itemNoun(kind, 1);
  switch (profile.inventory.implementation) {
    case "standard-mappings@1":
      return profile.standards.length === 0
        ? `${mode}, one ${unit} at a time.`
        : `${mode} against ${standardsText(profile)}, one ${unit} at a time.`;
    case "openapi-operations@1":
      return `${mode}, one endpoint of your API spec at a time.`;
    case "openapi-scans@1":
      return `${mode} of the endpoints your scan settings select.`;
    case "checklist@1":
      return `${mode}, one checklist item at a time.`;
    case "finding-candidates@1":
      return `${mode} of the finding candidates you supply.`;
    default:
      return `${mode}.`;
  }
}

export function presentCheckType(profile: AuditProfile): CheckTypePresentation {
  const known = Object.hasOwn(KNOWN, profile.ref.name)
    ? KNOWN[profile.ref.name]
    : undefined;
  return (
    known ?? {
      label: humanize(profile.ref.name),
      description: contractDescription(profile),
      order: 1000,
    }
  );
}

// ---------- Materials ----------

/** Whether a material's media type is one the input accepts ("format matches"). */
export function mediaTypeMatches(
  mediaType: string,
  accepted: readonly string[],
): boolean {
  return accepted.some((candidate) => {
    if (candidate === "*/*" || candidate === mediaType) return true;
    if (!candidate.endsWith("/*")) return false;
    return mediaType.startsWith(candidate.slice(0, -1));
  });
}

const INPUT_LABELS: Readonly<Record<string, string>> = {
  source: "Source code",
  sources: "Source code",
  openapi: "API spec",
  spec: "API spec",
  settings: "Scan settings",
  context: "Context brief",
  checklist: "Checklist",
};

const FORMAT_NAMES: Readonly<Record<string, string>> = {
  "application/zip": "ZIP",
  "application/x-zip-compressed": "ZIP",
  "application/json": "JSON",
  "application/yaml": "YAML",
  "application/x-yaml": "YAML",
  "text/yaml": "YAML",
  "text/plain": "text",
  "text/markdown": "Markdown",
  "text/x-markdown": "Markdown",
  "application/vnd.oai.openapi": "OpenAPI",
  "application/vnd.oai.openapi+json": "OpenAPI JSON",
  "application/vnd.oai.openapi+yaml": "OpenAPI YAML",
  "*/*": "any format",
  "text/*": "any text",
};

/** A label inside a sentence: "source code", but "API spec" keeps its case. */
export function inSentence(label: string): string {
  return /^[A-Z][A-Z]/.test(label)
    ? label
    : label.charAt(0).toLowerCase() + label.slice(1);
}

/** "Source code", "API spec"; other inputs by their name. */
export function inputLabel(name: string): string {
  return Object.hasOwn(INPUT_LABELS, name)
    ? INPUT_LABELS[name]!
    : humanize(name);
}

/** A media type in words: "ZIP", "JSON", or the media type itself. */
export function formatName(mediaType: string): string {
  return Object.hasOwn(FORMAT_NAMES, mediaType)
    ? FORMAT_NAMES[mediaType]!
    : mediaType;
}

/** "ZIP", "JSON or YAML", "JSON, YAML or ZIP". */
export function formatList(mediaTypes: readonly string[]): string {
  const names = [...new Set(mediaTypes.map(formatName))];
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} or ${names.at(-1)}`;
}

/** The project's current materials whose format matches an input. */
export function candidatesFor(
  input: ProfileInput,
  materials: readonly ArtifactMetadata[],
): ArtifactMetadata[] {
  return materials.filter((material) =>
    mediaTypeMatches(material.mediaType, input.mediaTypes),
  );
}

/** Inputs in a stable order: required first, then by name. */
export function orderedInputs(
  profile: AuditProfile,
): [name: string, input: ProfileInput][] {
  return Object.entries(profile.inputs).sort(
    ([leftName, left], [rightName, right]) =>
      Number(right.required) - Number(left.required) ||
      leftName.localeCompare(rightName),
  );
}

// ---------- Scope fields the check type passes to its workflows ----------

export type ScopeField = "objective" | "target" | "authorizationScope";

const SCOPE_FIELDS: readonly ScopeField[] = [
  "objective",
  "target",
  "authorizationScope",
];

/**
 * Scope fields the check type's workflows read (parameters mapped from a
 * `scope-field`). Only the profile detail lists workflows; without them the
 * set is empty.
 */
export function scopeFieldsUsed(profile: AuditProfile): Set<ScopeField> {
  const used = new Set<ScopeField>();
  for (const workflow of Object.values(profile.workflows ?? {})) {
    for (const parameter of Object.values(workflow.parameters)) {
      const name = parameter.name;
      if (
        parameter.source === "scope-field" &&
        name !== undefined &&
        (SCOPE_FIELDS as readonly string[]).includes(name)
      )
        used.add(name as ScopeField);
    }
  }
  return used;
}

// ---------- Readiness ----------

export type MissingNeed =
  /** No current material has a format this input takes. */
  | { kind: "input"; name: string; label: string; formats: string }
  /**
   * Inputs that compete for the same materials: each input needs a material
   * of its own, and `count` more are needed. Which input lacks one is not
   * known, so none is singled out.
   */
  | { kind: "shared"; names: string[]; labels: string[]; count: number }
  | { kind: "live-target" };

export type Readiness =
  | { state: "ready" }
  | { state: "missing"; missing: MissingNeed[] }
  | { state: "unavailable"; reasons: CompatibilityReason[] };

export interface ReadinessContext {
  /** The project's current materials read so far. */
  materials: readonly ArtifactMetadata[];
  /** The project has a live target in its settings. */
  hasLiveTarget: boolean;
  /** The check type's workflows read a target (from its profile detail). */
  needsLiveTarget: boolean;
}

/** The key that identifies one exact material in selects and choices. */
export function materialKey(
  material: Pick<ArtifactMetadata, "artifact">,
): string {
  return JSON.stringify([
    material.artifact.namespace,
    material.artifact.name,
    material.artifact.revision,
  ]);
}

/**
 * What required inputs lack when each needs a current material of its own
 * in a matching format: a maximum matching of inputs to materials (Kuhn's
 * augmenting paths). An input without any format match is missing on its
 * own. Inputs that only compete for too few materials are reported as one
 * group with the number of materials they lack.
 */
function missingInputs(
  inputs: readonly [name: string, input: ProfileInput][],
  materials: readonly ArtifactMetadata[],
): MissingNeed[] {
  const candidates = new Map(
    inputs.map(([name, input]) => [
      name,
      candidatesFor(input, materials).map(materialKey),
    ]),
  );
  const owner = new Map<string, string>();
  function assign(name: string, seen: Set<string>): boolean {
    for (const key of candidates.get(name) ?? []) {
      if (seen.has(key)) continue;
      seen.add(key);
      const current = owner.get(key);
      if (current === undefined || assign(current, seen)) {
        owner.set(key, name);
        return true;
      }
    }
    return false;
  }
  const unmatched = inputs
    .map(([name]) => name)
    .filter((name) => !assign(name, new Set()));

  const missing: MissingNeed[] = [];
  const grouped = new Set<string>();
  for (const name of unmatched) {
    const input = inputs.find(([candidate]) => candidate === name)?.[1];
    if (input === undefined || grouped.has(name)) continue;
    if ((candidates.get(name) ?? []).length === 0) {
      missing.push({
        kind: "input",
        name,
        label: inputLabel(name),
        formats: formatList(input.mediaTypes),
      });
      continue;
    }
    // Every material this input could take is taken; the inputs reachable
    // through those materials compete for the same pool.
    const group = new Set([name]);
    const queue = [name];
    for (let next = queue.shift(); next !== undefined; next = queue.shift()) {
      for (const key of candidates.get(next) ?? []) {
        const holder = owner.get(key);
        if (holder !== undefined && !group.has(holder)) {
          group.add(holder);
          queue.push(holder);
        }
      }
    }
    const names = inputs
      .map(([candidate]) => candidate)
      .filter((candidate) => group.has(candidate));
    for (const member of names) grouped.add(member);
    missing.push({
      kind: "shared",
      names,
      labels: names.map(inputLabel),
      count: unmatched.filter((candidate) => group.has(candidate)).length,
    });
  }
  return missing;
}

/**
 * Ready: the Server can run the check type, every required input can have a
 * current material of its own in a matching format, and a check type that
 * tests a live target finds one in the project settings. Formats are all
 * that is compared; the contents are not.
 */
export function readinessOf(
  profile: AuditProfile,
  context: ReadinessContext,
): Readiness {
  if (!profile.serverCompatible)
    return { state: "unavailable", reasons: profile.compatibilityReasons };
  const missing = missingInputs(
    orderedInputs(profile).filter(([, input]) => input.required),
    context.materials,
  );
  if (context.needsLiveTarget && !context.hasLiveTarget)
    missing.push({ kind: "live-target" });
  return missing.length === 0
    ? { state: "ready" }
    : { state: "missing", missing };
}

function joinOr(words: readonly string[]): string {
  return words.length <= 1
    ? (words[0] ?? "")
    : `${words.slice(0, -1).join(", ")} or ${words.at(-1)}`;
}

/**
 * "Source code (ZIP)", "Another material for API spec or scan settings",
 * "Live target URL".
 */
export function missingLabel(need: MissingNeed): string {
  switch (need.kind) {
    case "live-target":
      return "Live target URL";
    case "input":
      return `${need.label} (${need.formats})`;
    case "shared": {
      const labels = joinOr(need.labels.map(inSentence));
      return need.count === 1
        ? `Another material for ${labels}`
        : `${need.count} more materials for ${labels}`;
    }
  }
}

const REASON_WORDS: Readonly<Record<CompatibilityReason, string>> = {
  preparation_unsupported: "preparation steps",
  discovery_unsupported: "discovery steps",
  assessment_unsupported: "assessment steps",
  multiple_rounds_unsupported: "more than one round",
  batching_unsupported: "several items in one run",
  automatic_active_checks_unsupported: "automatic live tests",
  active_check_approval_unsupported: "approvals for live tests",
  finding_confirmation_unsupported: "confirming possible issues",
  manual_applicability_unsupported: "manual applicability decisions",
  report_acceptance_unsupported: "report acceptance",
  manual_item_unsupported: "manual items",
};

/** A Server compatibility reason in words. */
export function compatibilityReasonText(reason: CompatibilityReason): string {
  return Object.hasOwn(REASON_WORDS, reason)
    ? REASON_WORDS[reason]
    : reason.replaceAll("_", " ");
}

// ---------- What a check type covers ----------

export interface ScopeSummary {
  /** "70 requirements", "Up to 94 scenarios", "Every endpoint in the API spec". */
  readonly size: string;
  /** One quiet line about where the items come from. */
  readonly detail: string;
}

/** What the check type covers, from its inventory and limits. */
export function scopeSummary(profile: AuditProfile): ScopeSummary {
  const kind = checkItemKind(profile);
  const limit = profile.execution.maxItemsTotal;
  const selection = profile.inventory.standardSelection;
  if (selection !== undefined)
    return {
      size: itemCount(kind, selection.entryIds.length),
      // The selection's own scope text; its levels are in Advanced options.
      detail: selection.scope,
    };
  switch (profile.inventory.implementation) {
    case "standard-mappings@1":
      return {
        size: `Up to ${itemCount(kind, limit)}`,
        detail:
          profile.standards.length === 0
            ? "From the check type's standard."
            : `From ${standardsText(profile)}.`,
      };
    case "openapi-operations@1":
      return {
        size: "Every endpoint in the API spec",
        detail: `At most ${itemCount("endpoint", limit)}.`,
      };
    case "openapi-scans@1":
      return {
        size: "The endpoints your scan settings select",
        detail: `At most ${itemCount("endpoint", limit)}.`,
      };
    case "checklist@1":
      return {
        size: "Every item of your checklist",
        detail: `At most ${itemCount("item", limit)}.`,
      };
    default:
      return {
        size: `Up to ${itemCount(kind, limit)}`,
        detail: "From the materials you supply.",
      };
  }
}
