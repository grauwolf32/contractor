import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError, requireData } from "../../../api/error";
import type { components } from "../../../api/generated/public";
import { queryKeys } from "../../../api/query-keys";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { QueryView } from "../../../app/query-view";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";
import { Button, Field } from "../../../ui";
import {
  type ResearchMaterial,
  useResearchMaterials,
  useResearchProgram,
} from "./research-data";
import { CursorControls } from "../../../app/cursor-controls";

type Request = {
  key: string;
  revision: number;
  body: components["schemas"]["AuditTaskCreationRequest"];
};

function WorkDialog({ audit, onClose }: { audit: Audit; onClose: () => void }) {
  const api = usePublicAPI(),
    cache = useQueryClient();
  const [objective, setObjective] = useState("");
  const [method, setMethod] = useState("");
  const [inputs, setInputs] = useState<Record<string, ResearchMaterial>>({});
  const [locked, setLocked] = useState(false);
  const pending = useRef<Request | undefined>(undefined);
  const profile = useResearchProgram(audit);
  const materials = useResearchMaterials(audit);
  const program = profile.data?.program;
  const binding = program?.workflows[method];
  const slots = Object.entries(binding?.inputs ?? {}).filter(
    ([, mapping]) => mapping.source === "task-input",
  );
  const write = useMutation({
    mutationFn: async () => {
      let request = pending.current;
      if (!request) {
        const key = createMutationIdempotencyKey("audit-work");
        const selected: components["schemas"]["AuditTaskSelection"]["inputs"] =
          {};
        for (const [slot] of slots) {
          const selection = inputs[slot];
          if (!selection) continue;
          selected[slot] = selection.material;
        }
        request = {
          key,
          revision: audit.revision,
          body: {
            work: [{ key, workflow: method, objective, inputs: selected }],
            attention: [key],
          },
        };
        pending.current = request;
      }
      setLocked(true);
      return requireData(
        await api.request((client) =>
          client.POST("/v1/audits/{auditId}/tasks", {
            params: {
              path: { auditId: audit.auditId },
              header: {
                "Idempotency-Key": request.key,
                "If-Match": `"${request.revision}"`,
              },
            },
            body: request.body,
          }),
        ),
      );
    },
    onSuccess: async () => {
      pending.current = undefined;
      setLocked(false);
      await cache.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      });
      onClose();
    },
    onError: async (error) => {
      if (
        error instanceof PublicAPIError &&
        error.status >= 400 &&
        error.status < 500 &&
        ![408, 429].includes(error.status)
      ) {
        pending.current = undefined;
        setLocked(false);
      }
      await cache.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      });
    },
  });
  const close = () => {
    if (!write.isPending && !locked) onClose();
  };
  return (
    <Dialog
      className="ui-dialog ui-panel research-dialog"
      labelledBy="add-work-heading"
      onRequestClose={close}
    >
      <DialogHeader
        id="add-work-heading"
        title="Add work"
        eyebrow="Ordinary workflow · exact materials"
        close={{
          label: "Close work setup",
          disabled: write.isPending || locked,
          onClose: close,
        }}
      />
      <QueryView query={profile}>
        {() => (
          <form
            className="ui-stack"
            onSubmit={(e) => {
              e.preventDefault();
              write.mutate();
            }}
          >
            <Field label="What should this work investigate?" required>
              {(control) => (
                <textarea
                  {...control}
                  value={objective}
                  disabled={locked}
                  onChange={(e) => setObjective(e.target.value)}
                  rows={3}
                />
              )}
            </Field>
            <Field label="Method" required>
              {(control) => (
                <select
                  {...control}
                  value={method}
                  disabled={locked}
                  onChange={(e) => {
                    setMethod(e.target.value);
                    setInputs({});
                  }}
                >
                  <option value="">Choose from this program</option>
                  {Object.keys(program?.workflows ?? {}).map((alias) => (
                    <option key={alias} value={alias}>
                      {alias}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            {binding?.outputs.proposals ? (
              <p className="checks-notice">
                This workflow can propose hypotheses. Saving its output,
                registering hypotheses and accepting checking work are separate
                steps.
              </p>
            ) : null}
            {slots.length ? (
              <QueryView query={materials.query}>
                {(page) => (
                  <>
                    {slots.map(([slot]) => (
                      <Field
                        key={slot}
                        label={`Material for ${slot}`}
                        hint="Choose an exact retained material, including a material-set document when the method declares a set. The pinned input contract is validated by Server."
                      >
                        {(control) => (
                          <select
                            {...control}
                            value={inputs[slot]?.key ?? ""}
                            disabled={locked}
                            onChange={(e) =>
                              setInputs((previous) => {
                                const next = { ...previous };
                                const entry = page.items.find(
                                  (item) => item.key === e.target.value,
                                );
                                if (entry) next[slot] = entry;
                                else delete next[slot];
                                return next;
                              })
                            }
                          >
                            <option value="">Not selected</option>
                            {inputs[slot] &&
                            !page.items.some(
                              (item) => item.key === inputs[slot]?.key,
                            ) ? (
                              <option value={inputs[slot]!.key}>
                                {inputs[slot]!.material.ref.name}@
                                {inputs[slot]!.material.ref.revision} · selected
                                exact revision
                              </option>
                            ) : null}
                            {page.items.map((entry) => (
                              <option key={entry.key} value={entry.key}>
                                {entry.material.ref.name}@
                                {entry.material.ref.revision} ·{" "}
                                {entry.material.media_type}
                              </option>
                            ))}
                          </select>
                        )}
                      </Field>
                    ))}
                    <CursorControls
                      label="Work material pages"
                      {...materials.pages.controls(page.page)}
                    />
                  </>
                )}
              </QueryView>
            ) : (
              <p className="ui-caption">
                Configured exact Audit inputs and system context are attached
                automatically. This work does not require a manually authored
                hypothesis.
              </p>
            )}
            {write.error ? (
              <ErrorNotice error={write.error} reconcileWrite />
            ) : null}
            {locked && write.error ? (
              <p className="ui-caption">
                The acknowledgement is uncertain. Retry the retained request to
                reconcile it; no new work identity is created.
              </p>
            ) : null}
            <div className="ui-dialog-actions">
              <Button disabled={write.isPending || locked} onClick={close}>
                Cancel
              </Button>
              <Button
                type="submit"
                variant="primary"
                pending={write.isPending}
                disabled={!method || !objective.trim()}
              >
                {locked ? "Retry accepting work" : "Accept work"}
              </Button>
            </div>
          </form>
        )}
      </QueryView>
    </Dialog>
  );
}

export function AddResearchWork({ audit }: { audit: Audit }) {
  const [open, setOpen] = useState(false);
  if (audit.state !== "active" && audit.state !== "paused") return null;
  return (
    <>
      <Button size="sm" variant="primary" onClick={() => setOpen(true)}>
        Add work
      </Button>
      {open ? (
        <WorkDialog audit={audit} onClose={() => setOpen(false)} />
      ) : null}
    </>
  );
}
