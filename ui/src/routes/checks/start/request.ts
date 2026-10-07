/**
 * The create request (POST /v1/projects/{projectId}/audits) built from the
 * Start form: the exact check type version, one exact material per chosen
 * input, the non-empty scope fields and the runtime labels.
 */
import type {
  AuditProfile,
  CreateAuditRequest,
  ExactArtifactRef,
} from "../../../api/audits";

/** Longest objective, target and authorization scope the form accepts. */
export const SCOPE_TEXT_LIMIT = 4096;

// RuntimeInfrastructureId in the public API: lowercase, at most 63 bytes.
const RUNTIME_LABEL = /^[a-z][a-z0-9_-]{0,62}$/;
export const MAX_RUNTIME_LABELS = 32;

export interface ParsedRuntimeLabels {
  /** Unique labels in the order typed. */
  labels: string[];
  /** Typed values that are not valid runtime labels. */
  invalid: string[];
}

/** Labels separated by commas or spaces, as in "debug, caido". */
export function parseRuntimeLabels(text: string): ParsedRuntimeLabels {
  const values = [
    ...new Set(
      text
        .split(/[\s,]+/u)
        .map((value) => value.trim())
        .filter((value) => value !== ""),
    ),
  ];
  return {
    labels: values.filter((value) => RUNTIME_LABEL.test(value)),
    invalid: values.filter((value) => !RUNTIME_LABEL.test(value)),
  };
}

export interface CreateRequestDraft {
  profile: Pick<AuditProfile["ref"], "name" | "version">;
  /** Chosen material per input name. */
  inputs: Readonly<Record<string, ExactArtifactRef>>;
  objective: string;
  target: string;
  authorizationScope: string;
  runtimeLabels: readonly string[];
}

export function createRequest(draft: CreateRequestDraft): CreateAuditRequest {
  const scope = Object.fromEntries(
    (
      [
        ["objective", draft.objective],
        ["target", draft.target],
        ["authorizationScope", draft.authorizationScope],
      ] as const
    )
      .map(([field, value]) => [field, value.trim()] as const)
      .filter(([, value]) => value !== ""),
  );
  return {
    profile: { name: draft.profile.name, version: draft.profile.version },
    inputs: Object.fromEntries(
      Object.entries(draft.inputs).map(([name, ref]) => [
        name,
        { namespace: ref.namespace, name: ref.name, revision: ref.revision },
      ]),
    ),
    ...(draft.runtimeLabels.length === 0
      ? {}
      : { runtimeLabels: [...draft.runtimeLabels] }),
    ...(Object.keys(scope).length === 0 ? {} : { scope }),
  };
}
