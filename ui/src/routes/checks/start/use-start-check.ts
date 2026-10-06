/**
 * State of the Start page: which check type is chosen (the `type` query
 * parameter, else the suggestion for the objective the page opened with,
 * else the first ready check type), the form, and what still keeps the
 * check from starting.
 */
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import type { ArtifactMetadata } from "../../../api/artifacts";
import type {
  AuditProfile,
  CreateAuditRequest,
  ExactArtifactRef,
} from "../../../api/audits";
import type { Project } from "../../../api/projects";
import {
  candidatesFor,
  inSentence,
  inputLabel,
  formatList,
  materialKey,
  orderedInputs,
  presentCheckType,
  readinessOf,
  scopeFieldsUsed,
  type CheckTypeFamily,
  type CheckTypePresentation,
  type CheckTypeRow,
  type ProfileInput,
  type Readiness,
  type ScopeField,
} from "./check-types";
import {
  useCheckTypeCatalog,
  useProfileDetail,
  useProject,
  useProjectMaterials,
  versionKey,
  type CheckTypeCatalog,
  type ProjectMaterials,
} from "./data";
import { useLaunch, type Launch } from "./launch";
import {
  createRequest,
  MAX_RUNTIME_LABELS,
  parseRuntimeLabels,
  type ParsedRuntimeLabels,
} from "./request";
import { suggestCheckTypes, type SuggestionResult } from "./suggestions";
import {
  DEFAULT_TIME_LIMIT,
  timeLimitSeconds,
  type TimeLimitState,
} from "./time-limit";

export type ReadinessGroup = "ready" | "missing" | "unavailable" | "unknown";

export interface RowView {
  readonly key: string;
  readonly row: CheckTypeRow;
  /** The variant the row opens: the chosen one when selected. */
  readonly family: CheckTypeFamily;
  readonly presentation: CheckTypePresentation;
  /** Undefined while the project's materials could not be read. */
  readonly readiness: Readiness | undefined;
  readonly group: ReadinessGroup;
  /** What it takes: required inputs and a live target. */
  readonly uses: readonly string[];
  readonly href: string;
  readonly selected: boolean;
  readonly suggested: boolean;
}

export interface InputSlot {
  readonly name: string;
  readonly input: ProfileInput;
  readonly label: string;
  readonly formats: string;
  /** Current materials whose format matches. */
  readonly candidates: readonly ArtifactMetadata[];
  /** The chosen material's key, "" for none. */
  readonly selection: string;
  readonly selected: ArtifactMetadata | undefined;
  /** Chosen by the page: the only format match of a complete inventory. */
  readonly automatic: boolean;
  /** The user chose this slot's value (including clearing it). */
  readonly explicit: boolean;
}

export interface Selection {
  readonly row: RowView;
  readonly family: CheckTypeFamily;
  /** The version the form submits (from the list). */
  readonly profile: AuditProfile;
  readonly presentation: CheckTypePresentation;
}

function hrefWith(
  params: URLSearchParams,
  changes: Readonly<Record<string, string | undefined>>,
): string {
  const next = new URLSearchParams(params);
  for (const [name, value] of Object.entries(changes)) {
    if (value === undefined) next.delete(name);
    else next.set(name, value);
  }
  const query = next.toString();
  return query === "" ? "/checks/new" : `/checks/new?${query}`;
}

function groupOf(readiness: Readiness | undefined): ReadinessGroup {
  return readiness === undefined
    ? "unknown"
    : readiness.state === "ready"
      ? "ready"
      : readiness.state;
}

/** The variant a row opens by default: the widest one the Server can run. */
function defaultVariant(row: CheckTypeRow): CheckTypeFamily {
  return (
    row.variants.find((family) => family.preferred.serverCompatible) ??
    row.primary
  );
}

function usesOf(profile: AuditProfile, needsLiveTarget: boolean): string[] {
  return [
    ...orderedInputs(profile)
      .filter(([, input]) => input.required)
      .map(([name]) => inputLabel(name)),
    ...(needsLiveTarget ? ["Live target"] : []),
  ];
}

const NO_CHOICES: Readonly<Record<string, string>> = {};

export interface StartCheck {
  readonly projectId: string;
  readonly project: Project | undefined;
  readonly href: (
    changes: Readonly<Record<string, string | undefined>>,
  ) => string;
  readonly catalog: CheckTypeCatalog;
  readonly materials: ProjectMaterials;
  /** Check types and readiness can be listed. */
  readonly listReady: boolean;
  readonly rows: readonly RowView[];
  /** Rows in list order (ready, then missing, unavailable or unknown). */
  readonly ordered: readonly RowView[];
  /** The `type` parameter names no published check type. */
  readonly unknownType: string | undefined;
  readonly selection: Selection | undefined;
  /** Whether the `type` parameter chose the selection. */
  readonly typeChosen: boolean;
  readonly detail: ReturnType<typeof useProfileDetail>;
  /** The exact version detail, once loaded. */
  readonly exact: AuditProfile | undefined;
  readonly readiness: Readiness | undefined;
  readonly objective: string;
  readonly setObjective: (value: string) => void;
  readonly suggestions: SuggestionResult;
  readonly isReady: (checkType: string) => boolean;
  readonly slots: readonly InputSlot[];
  readonly chooseMaterial: (input: string, value: string) => void;
  readonly scopeFields: ReadonlySet<ScopeField>;
  readonly target: string;
  readonly setTarget: (value: string) => void;
  readonly authorizationScope: string;
  readonly setAuthorizationScope: (value: string) => void;
  readonly runtimeLabels: string;
  readonly setRuntimeLabels: (value: string) => void;
  readonly parsedLabels: ParsedRuntimeLabels;
  readonly timeLimit: TimeLimitState;
  readonly setTimeLimit: (value: TimeLimitState) => void;
  readonly deadlineSeconds: number | undefined;
  readonly chooseVersion: (version: string) => void;
  /** The create request, when the form is complete. */
  readonly request: CreateAuditRequest | undefined;
  /** What keeps the check from being created ("Choose the …"). */
  readonly draftBlocker: string | undefined;
  /** What keeps the check from starting. */
  readonly startBlocker: string | undefined;
  readonly launch: Launch;
  readonly start: () => void;
  readonly saveDraft: () => void;
}

export function useStartCheck(
  projectId: string,
  initialObjective: string,
): StartCheck {
  const [params] = useSearchParams();
  const typeParam = params.get("type") ?? undefined;
  const projectQuery = useProject(projectId);
  const project = projectQuery.data;
  const catalog = useCheckTypeCatalog();
  const materials = useProjectMaterials(projectId);
  const launch = useLaunch(projectId);

  const [objective, setObjective] = useState(initialObjective);
  const [versions, setVersions] = useState<Readonly<Record<string, string>>>(
    {},
  );
  const [choices, setChoices] = useState<{
    contract: string;
    values: Readonly<Record<string, string>>;
  }>({ contract: "", values: NO_CHOICES });
  const [target, setTarget] = useState<string>();
  const [authorizationScope, setAuthorizationScope] = useState("");
  const [runtimeLabels, setRuntimeLabels] = useState("");
  const [timeLimit, setTimeLimit] = useState<TimeLimitState>({
    choice: DEFAULT_TIME_LIMIT,
    hours: "24",
  });

  const hasLiveTarget = project?.httpTarget !== undefined;
  const materialItems = materials.items;

  // Readiness of every family's preferred version.
  const familyReadiness = useMemo(() => {
    const result = new Map<string, Readiness | undefined>();
    for (const family of catalog.families) {
      const detail = catalog.details.get(versionKey(family.preferred));
      result.set(
        family.name,
        materialItems === undefined
          ? undefined
          : readinessOf(family.preferred, {
              materials: materialItems,
              hasLiveTarget,
              needsLiveTarget:
                detail !== undefined && scopeFieldsUsed(detail).has("target"),
            }),
      );
    }
    return result;
  }, [catalog.families, catalog.details, materialItems, hasLiveTarget]);

  const isReady = useMemo(
    () => (checkType: string) =>
      familyReadiness.get(checkType)?.state === "ready",
    [familyReadiness],
  );
  const suggestions = useMemo(
    () => suggestCheckTypes(objective, isReady),
    [objective, isReady],
  );
  const opening = useMemo(
    () => suggestCheckTypes(initialObjective, isReady),
    [initialObjective, isReady],
  );

  const familyByName = useMemo(
    () => new Map(catalog.families.map((family) => [family.name, family])),
    [catalog.families],
  );
  const requested =
    typeParam === undefined ? undefined : familyByName.get(typeParam);

  // Rows with their readiness, before the selection is known.
  const baseRows = useMemo(
    () =>
      catalog.rows.map((row) => {
        const family = defaultVariant(row);
        const readiness = familyReadiness.get(family.name);
        const detail = catalog.details.get(versionKey(family.preferred));
        return {
          row,
          family,
          readiness,
          group: groupOf(readiness),
          uses: usesOf(
            family.preferred,
            detail !== undefined && scopeFieldsUsed(detail).has("target"),
          ),
        };
      }),
    [catalog.rows, catalog.details, familyReadiness],
  );
  const groupOrder: readonly ReadinessGroup[] = [
    "ready",
    "missing",
    "unavailable",
    "unknown",
  ];
  const orderedBase = groupOrder.flatMap((group) =>
    baseRows.filter((row) => row.group === group),
  );
  const fallbackName =
    opening.suggestion?.checkType ?? orderedBase[0]?.family.name;
  const selectedName = requested?.name ?? fallbackName;

  const rows: RowView[] = orderedBase.map((base) => {
    const chosen = base.row.variants.find(
      (family) => family.name === selectedName,
    );
    const family = chosen ?? base.family;
    return {
      key: base.row.primary.name,
      row: base.row,
      family,
      presentation: presentCheckType(base.row.primary.preferred),
      readiness: base.readiness,
      group: base.group,
      uses: base.uses,
      href: hrefWith(params, { type: family.name }),
      selected: chosen !== undefined,
      suggested: base.row.variants.some(
        (variant) => variant.name === suggestions.suggestion?.checkType,
      ),
    };
  });
  const selectedRow = rows.find((row) => row.selected);
  const selectedFamily = selectedRow?.family;
  const selectedVersion =
    selectedFamily === undefined
      ? undefined
      : (versions[selectedFamily.name] ?? selectedFamily.preferred.ref.version);
  const selectedProfile =
    selectedFamily?.versions.find(
      (profile) => profile.ref.version === selectedVersion,
    ) ?? selectedFamily?.preferred;
  const selection: Selection | undefined =
    selectedRow === undefined ||
    selectedFamily === undefined ||
    selectedProfile === undefined
      ? undefined
      : {
          row: selectedRow,
          family: selectedFamily,
          profile: selectedProfile,
          presentation: presentCheckType(selectedProfile),
        };

  const detail = useProfileDetail(selectedProfile);
  const exact =
    detail.data !== undefined &&
    selectedProfile !== undefined &&
    versionKey(detail.data) === versionKey(selectedProfile)
      ? detail.data
      : undefined;
  const contractProfile = exact ?? selectedProfile;
  const scopeFields = useMemo(
    () =>
      exact === undefined ? new Set<ScopeField>() : scopeFieldsUsed(exact),
    [exact],
  );
  const usesTarget = scopeFields.has("target");
  const readiness =
    selectedProfile === undefined || materialItems === undefined
      ? undefined
      : readinessOf(selectedProfile, {
          materials: materialItems,
          hasLiveTarget,
          needsLiveTarget: usesTarget,
        });

  // Explicit material choices last while the input contract stays the same.
  const contract =
    contractProfile === undefined
      ? ""
      : JSON.stringify(
          orderedInputs(contractProfile).map(([name, input]) => [
            name,
            input.required,
            input.mediaTypes,
          ]),
        );
  const explicitChoices =
    choices.contract === contract ? choices.values : NO_CHOICES;
  const contractInputs =
    contractProfile === undefined ? [] : orderedInputs(contractProfile);
  // A slot gets its only format match automatically, once every page is
  // read. A material that is the only match of several slots is ambiguous
  // and waits for a choice.
  const soleMatches = new Map<string, number>();
  if (materials.complete) {
    for (const [name, input] of contractInputs) {
      if (Object.hasOwn(explicitChoices, name)) continue;
      const candidates = candidatesFor(input, materialItems ?? []);
      if (candidates.length !== 1) continue;
      const key = materialKey(candidates[0]!);
      soleMatches.set(key, (soleMatches.get(key) ?? 0) + 1);
    }
  }
  const slots: InputSlot[] = contractInputs.map(([name, input]) => {
    const candidates = candidatesFor(input, materialItems ?? []);
    const explicit = Object.hasOwn(explicitChoices, name);
    const only =
      materials.complete &&
      candidates.length === 1 &&
      soleMatches.get(materialKey(candidates[0]!)) === 1
        ? candidates[0]
        : undefined;
    const selectionKey = explicit
      ? (explicitChoices[name] ?? "")
      : only === undefined
        ? ""
        : materialKey(only);
    return {
      name,
      input,
      label: inputLabel(name),
      formats: formatList(input.mediaTypes),
      candidates,
      selection: selectionKey,
      selected: candidates.find(
        (candidate) => materialKey(candidate) === selectionKey,
      ),
      automatic: !explicit && only !== undefined,
      explicit,
    };
  });

  const effectiveTarget =
    target ?? (usesTarget ? (project?.httpTarget?.url ?? "") : "");
  const parsedLabels = useMemo(
    () => parseRuntimeLabels(runtimeLabels),
    [runtimeLabels],
  );
  const deadlineSeconds = timeLimitSeconds(timeLimit);

  // What keeps the check from being created, most basic first.
  let draftBlocker: string | undefined;
  const missingSlot = slots.find(
    (slot) => slot.input.required && slot.selected === undefined,
  );
  if (selection === undefined) draftBlocker = "Choose a check type.";
  else if (project?.lifecycle === "deleting")
    draftBlocker = "This project is being deleted.";
  else if (!selection.profile.serverCompatible)
    draftBlocker = "This check type can't run on this server.";
  else if (exact === undefined)
    draftBlocker =
      detail.error === null
        ? "Loading the check type…"
        : "The check type could not be loaded.";
  else if (materialItems === undefined)
    draftBlocker =
      materials.error === null
        ? "Loading the project's materials…"
        : "The project's materials could not be loaded.";
  else if (materials.loading) draftBlocker = "Loading the project's materials…";
  else if (missingSlot !== undefined)
    draftBlocker =
      missingSlot.candidates.length === 0
        ? `Add ${inSentence(missingSlot.label)} (${missingSlot.formats}) to the project first.`
        : `Choose the ${inSentence(missingSlot.label)} material.`;
  else if (scopeFields.has("objective") && objective.trim() === "")
    draftBlocker = "Write your objective; this check type uses it.";
  else if (usesTarget && effectiveTarget.trim() === "")
    draftBlocker = "Enter the target to test.";
  else if (
    scopeFields.has("authorizationScope") &&
    authorizationScope.trim() === ""
  )
    draftBlocker = "Describe the authorization scope.";
  else if (parsedLabels.invalid.length > 0)
    draftBlocker = "Fix the runtime labels.";
  else if (parsedLabels.labels.length > MAX_RUNTIME_LABELS)
    draftBlocker = `Use at most ${MAX_RUNTIME_LABELS} runtime labels.`;

  let request: CreateAuditRequest | undefined;
  if (draftBlocker === undefined && exact !== undefined) {
    const inputs: Record<string, ExactArtifactRef> = {};
    for (const slot of slots) {
      if (slot.selected !== undefined)
        inputs[slot.name] = slot.selected.artifact;
    }
    request = createRequest({
      profile: exact.ref,
      inputs,
      objective,
      target: effectiveTarget,
      authorizationScope,
      runtimeLabels: parsedLabels.labels,
    });
  }
  const startBlocker =
    draftBlocker ??
    (deadlineSeconds === undefined
      ? "Enter a time limit from 0.01 to 8760 hours."
      : undefined);

  function chooseMaterial(input: string, value: string): void {
    setChoices({
      contract,
      values: { ...explicitChoices, [input]: value },
    });
  }

  return {
    projectId,
    project,
    href: (changes) => hrefWith(params, changes),
    catalog,
    materials,
    listReady:
      catalog.query.isSuccess &&
      catalog.detailsSettled &&
      projectQuery.status !== "pending" &&
      (materialItems !== undefined
        ? !materials.loading
        : materials.error !== null),
    rows,
    ordered: rows,
    unknownType:
      typeParam !== undefined &&
      catalog.query.isSuccess &&
      requested === undefined
        ? typeParam
        : undefined,
    selection,
    typeChosen: requested !== undefined,
    detail,
    exact,
    readiness,
    objective,
    setObjective,
    suggestions,
    isReady,
    slots,
    chooseMaterial,
    scopeFields,
    target: effectiveTarget,
    setTarget,
    authorizationScope,
    setAuthorizationScope,
    runtimeLabels,
    setRuntimeLabels,
    parsedLabels,
    timeLimit,
    setTimeLimit,
    deadlineSeconds,
    chooseVersion: (version) => {
      if (selectedFamily === undefined) return;
      setVersions((current) => ({
        ...current,
        [selectedFamily.name]: version,
      }));
    },
    request,
    draftBlocker,
    startBlocker,
    launch,
    start: () => {
      if (request !== undefined && deadlineSeconds !== undefined)
        launch.launch("start", { request, deadlineSeconds });
    },
    saveDraft: () => {
      if (request !== undefined)
        launch.launch("draft", {
          request,
          deadlineSeconds: deadlineSeconds ?? 0,
        });
    },
  };
}
