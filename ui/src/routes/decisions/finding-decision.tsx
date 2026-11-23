import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, useRef, useLayoutEffect } from "react";
import {
  editProjectFinding,
  type ProjectFinding,
  type ProjectFindingEditResult,
  type FindingVerdict,
  type FindingSeverity,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { Button, DecisionBar } from "../../ui";
import {
  findingDecisionBody,
  findingDraftProblem,
  MAX_RATIONALE_BYTES,
  SEVERITY_OPTIONS,
  VERDICT_ACTIONS,
} from "./model";
import { aiSummary } from "./text";
import { DuplicatePicker } from "./duplicate-picker";
import "./decisions.css";

type EditOptions = Parameters<typeof editProjectFinding>[1];
interface Props {
  presentation?: "review" | undefined;
  finding: ProjectFinding;
  onDecided?: ((result: ProjectFindingEditResult) => void) | undefined;
  onRecording?: ((pending: boolean) => void) | undefined;
  next?: { label: string; onNext: () => void } | undefined;
  autoFocus?: boolean | undefined;
  shortcuts?: boolean | undefined;
}
function Editor({
  finding,
  onDecided,
  onRecording,
  next,
  shortcuts = true,
  presentation,
  autoFocus = false,
}: Props) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const root = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    if (autoFocus)
      root.current?.querySelector<HTMLButtonElement>("button")?.focus();
  }, [autoFocus]);
  const [baseRevision, setBaseRevision] = useState(finding.revision);
  const [verdict, setVerdict] = useState<FindingVerdict | undefined>(undefined);
  const [severity, setSeverity] = useState<FindingSeverity | undefined>(
    finding.analystSeverity ?? undefined,
  );
  const [rationale, setRationale] = useState("");
  const [target, setTarget] = useState("");
  const [notice, setNotice] = useState("");
  const [editing, setEditing] = useState(false);
  const wasEditing = useRef(false);
  useLayoutEffect(() => {
    if (wasEditing.current !== editing)
      root.current?.querySelector<HTMLButtonElement>("button")?.focus();
    wasEditing.current = editing;
  }, [editing]);
  const review = presentation === "review";
  const [keys] = useState(
    () => new MutationDraftKeyring<EditOptions>("project-finding"),
  );
  const refresh = async () => {
    await Promise.all([
      client.resetQueries({
        queryKey: ["project-findings", finding.projectId],
        predicate: (query) =>
          query.queryKey.includes("list") ||
          query.queryKey.includes("duplicate-targets"),
      }),

      client.invalidateQueries({
        queryKey: ["project-findings", finding.projectId],
        predicate: (query) =>
          !query.queryKey.includes("sources") &&
          !query.queryKey.includes("list") &&
          !query.queryKey.includes("duplicate-targets"),
      }),
      client.invalidateQueries({
        queryKey: ["audits"],
        predicate: (query) =>
          query.queryKey.includes("findings") ||
          query.queryKey.includes("finding") ||
          query.queryKey.includes("workspace"),
      }),
      client.invalidateQueries({
        queryKey: ["cross-project", "findings", "owner"],
      }),
    ]);
  };
  const mutation = useMutation({
    mutationFn: (options: EditOptions) => editProjectFinding(api, options),
    onSuccess: async (result) => {
      setBaseRevision(result.finding.revision);
      setNotice("Assessment saved.");
      setRationale("");
      setVerdict(undefined);
      setEditing(false);
      await refresh();
      onDecided?.(result);
    },
    onError: async () => {
      await refresh();
    },
    onSettled: () => {
      onRecording?.(false);
    },
  });
  const collapsed =
    review &&
    finding.analystVerdict !== null &&
    !editing &&
    verdict === undefined &&
    !mutation.isPending;
  const draft =
    verdict === undefined
      ? undefined
      : { verdict, severity, rationale, duplicateTargetId: target };
  const problem =
    draft === undefined
      ? undefined
      : findingDraftProblem(draft, finding.findingId);
  const stale = finding.revision !== baseRevision && !mutation.isPending;
  useLayoutEffect(() => {
    if (notice !== "" && !mutation.isPending)
      root.current
        ?.querySelector<HTMLButtonElement>("button:not(:disabled)")
        ?.focus();
  }, [notice, mutation.isPending]);
  const submit = () => {
    if (
      draft === undefined ||
      problem !== undefined ||
      stale ||
      mutation.isPending
    )
      return;
    const decision = findingDecisionBody(draft);
    const base = {
      projectId: finding.projectId,
      findingId: finding.findingId,
      expectedRevision: baseRevision,
      decision,
      idempotencyKey: "",
    };
    const options = { ...base, idempotencyKey: keys.keyFor(base) };
    onRecording?.(true);
    setNotice("");
    mutation.mutate(options);
  };
  const summaryAction =
    finding.claim?.description && (!review || verdict !== undefined) ? (
      <Button
        size="xs"
        disabled={mutation.isPending || stale}
        onClick={() => setRationale(aiSummary(finding))}
      >
        Use issue summary
      </Button>
    ) : undefined;
  return (
    <section
      ref={root}
      aria-label="Assess possible issue"
      className="decisions-finding"
      data-review-layout={presentation === "review" || undefined}
    >
      {notice === "" ? null : (
        <p className="decisions-save-notice" role="status">
          {notice}
        </p>
      )}
      {collapsed ? (
        <div className="decisions-review-actions">
          <Button size="sm" onClick={() => setEditing(true)}>
            Change decision
          </Button>
        </div>
      ) : (
        <>
          {stale ? (
            <p role="alert">
              This possible issue changed. Your draft is still here.{" "}
              <Button
                size="xs"
                onClick={() => {
                  setBaseRevision(finding.revision);
                  mutation.reset();
                }}
              >
                Use current revision
              </Button>
            </p>
          ) : null}
          <DecisionBar
            deferDraft={presentation === "review"}
            options={(Object.keys(VERDICT_ACTIONS) as FindingVerdict[])
              .filter(
                (id) =>
                  id !== "reopen" ||
                  finding.analystVerdict !== null ||
                  verdict === "reopen",
              )
              .map((id) => ({
                id,
                label:
                  presentation === "review" && id === "duplicate"
                    ? "Mark as duplicate"
                    : VERDICT_ACTIONS[id],
                ...(shortcuts
                  ? {
                      shortcut: {
                        true_positive: "c",
                        false_positive: "r",
                        needs_evidence: "e",
                        duplicate: "d",
                        reopen: "o",
                      }[id],
                    }
                  : {}),
              }))}
            more={
              review && editing ? (
                <>
                  {summaryAction}
                  <Button
                    size="xs"
                    disabled={mutation.isPending}
                    onClick={() => {
                      setEditing(false);
                      setVerdict(undefined);
                      setRationale("");
                      setNotice("");
                    }}
                  >
                    Cancel
                  </Button>
                </>
              ) : (
                summaryAction
              )
            }
            selected={verdict}
            onSelect={(value) => {
              setVerdict(value as FindingVerdict);
              setNotice("");
            }}
            severity={
              verdict === "true_positive"
                ? {
                    options: SEVERITY_OPTIONS,
                    value: severity,
                    onChange: (value) => setSeverity(value as FindingSeverity),
                    required: true,
                  }
                : undefined
            }
            rationale={{
              value: rationale,
              onChange: setRationale,
              maxLength: MAX_RATIONALE_BYTES,
              placeholder: "What led you to this decision?",
              hint: "Required. Saved with your assessment.",
            }}
            onSubmit={submit}
            submitLabel="Record decision"
            pending={mutation.isPending}
            next={presentation === "review" ? undefined : next}
            disabledReason={
              stale ? "Review the current revision before saving." : undefined
            }
            validationProblem={problem}
            extra={
              verdict === "duplicate" ? (
                <DuplicatePicker
                  projectId={finding.projectId}
                  findingId={finding.findingId}
                  value={target}
                  onChange={setTarget}
                />
              ) : undefined
            }
            error={
              mutation.error === null ? undefined : (
                <span>
                  {mutation.error.message}{" "}
                  {mutation.variables === undefined ? null : (
                    <Button
                      size="xs"
                      onClick={() => {
                        onRecording?.(true);
                        mutation.mutate(mutation.variables!);
                      }}
                    >
                      Retry this save
                    </Button>
                  )}
                </span>
              )
            }
          />
        </>
      )}
    </section>
  );
}
export function FindingDecision(props: Props) {
  return <Editor key={props.finding.findingId} {...props} />;
}
