import { isWebScanProfile } from "./web-scan-contract";
/**
 * Check types (AuditProfiles) as the Start page shows them: families of
 * versions, rows that fold scope variants together, plain-language labels,
 * and readiness against the project's current materials.
 *
 * Everything here is derived from the profile contract the Server publishes
 * (inputs, policy, limits and pinned workflows) and from the
 * project's current materials. Media types decide compatibility: a match
 * means the format matches, never that a material is the right one
 * (docs/spec/ui/journeys.md US-03).
 */
import type { IconName } from "../../../app/icon";
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { AuditProfile } from "../../../api/audits";
import { compareVersions } from "../../../app/format";
import { checkItemKind, itemCount } from "../../../app/vocabulary";

export type ProfileInput = AuditProfile["inputs"][string];
type CompatibilityReason = string;

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
    const preferred = sorted[0]!;
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

/** Rows in display order: known check types first, then by label. */
export function groupRows(
  families: readonly CheckTypeFamily[],
): CheckTypeRow[] {
  return families
    .map((family) => ({ primary: family, variants: [family] }))
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
  "web-scan": {
    label: "Web scan",
    description:
      "Checks a project URL with installed Nuclei rules or your custom YAML or ZIP rulepack.",
    order: 65,
  },
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
  "openapi-nuclei-scan": {
    label: "Nuclei scan",
    description:
      "Runs Nuclei on the endpoints your scan settings select, using your project’s active-check setting.",
    order: 70,
  },
  "source-openapi-sqlmap-scan": {
    label: "Source code to SQLMap checks",
    description:
      "Prepares a validated API spec from source code, accepts its selected endpoints, then runs request checks using your scan settings.",
    order: 81,
  },
  "openapi-sqlmap-scan": {
    label: "SQLMap scan",
    description:
      "Runs SQLMap on the endpoints your scan settings select, using your project’s active-check setting.",
    order: 80,
  },
};

function humanize(name: string): string {
  const words = name.replaceAll(/[-_.]+/g, " ").trim();
  return words === "" ? name : words.charAt(0).toUpperCase() + words.slice(1);
}

function contractDescription(profile: AuditProfile): string {
  return `Runs accepted work against selected materials, up to ${profile.limits.max_tasks} tasks.`;
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
  engagement: "Engagement",
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

/** "a", "a and b", "a, b and c"; `or` joins alternatives instead. */
export function joinWords(
  words: readonly string[],
  conjunction: "and" | "or" = "and",
): string {
  return words.length <= 1
    ? (words[0] ?? "")
    : `${words.slice(0, -1).join(", ")} ${conjunction} ${words.at(-1)}`;
}

/** "ZIP", "JSON or YAML", "JSON, YAML or ZIP". */
export function formatList(mediaTypes: readonly string[]): string {
  return joinWords([...new Set(mediaTypes.map(formatName))], "or");
}

/** The project's current materials whose format matches an input. */
export function candidatesFor(
  input: ProfileInput,
  materials: readonly ArtifactMetadata[],
): ArtifactMetadata[] {
  return materials.filter((material) =>
    mediaTypeMatches(material.mediaType, input.media_types),
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

/** Scope annotations are optional; executable targets come from exact inputs. */
export function scopeFieldsUsed(profile: AuditProfile): Set<ScopeField> {
  void profile;
  return new Set();
}

// ---------- Readiness ----------

type MissingNeed =
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

interface ReadinessContext {
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
        formats: formatList(input.media_types),
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
  if (isWebScanProfile(profile)) return { state: "ready" };
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

// ---------- What a check type covers ----------

interface ScopeSummary {
  /** "70 requirements", "Up to 94 scenarios", "Every endpoint in the API spec". */
  readonly size: string;
  /** One quiet line about where the items come from. */
  readonly detail: string;
}

/** What the check type covers, from its inventory and limits. */
export function scopeSummary(profile: AuditProfile): ScopeSummary {
  return {
    size: `Up to ${itemCount(checkItemKind(profile), profile.limits.max_tasks)}`,
    detail:
      "Accepted tasks use the selected exact materials. Preparation may add endpoint tasks within this limit.",
  };
}

/** Common programs use one check glyph; their inputs describe the domain. */
export function checkTypeIcon(): IconName {
  return "checks";
}
