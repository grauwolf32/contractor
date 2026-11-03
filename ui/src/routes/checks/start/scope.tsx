import { useId, useState } from "react";
import { Link, useNavigate } from "react-router";
import { ErrorNotice } from "../../../app/error-notice";
import { AUTOMATIC_DOCUMENT_BYTES } from "../../../app/document-limits";
import { Button, Facts } from "../../../ui";
import { artifactDetailPath } from "../../artifacts/paths";
import { presentCheckType, scopeSummary } from "./check-types";
import { MaterialSlot } from "./materials";
import { projectPaths } from "./paths";
import type { StartCheck } from "./use-start-check";

export function ScopeChoices({ model }: { model: StartCheck }) {
  const id = useId();
  const navigate = useNavigate();
  const { selection } = model;
  if (selection === undefined || selection.row.row.variants.length <= 1)
    return null;
  return (
    <fieldset className="start-review-coverage">
      <legend>Review coverage</legend>
      {selection.row.row.variants.map((family) => {
        const summary = scopeSummary(family.preferred);
        return (
          <label key={family.name} className="start-scope-option">
            <input
              type="radio"
              name={`${id}-coverage`}
              checked={family.name === selection.family.name}
              onChange={() =>
                void navigate(model.href({ type: family.name }), {
                  replace: true,
                })
              }
            />
            <span>
              <span className="start-scope-size">{summary.size}</span>
              <span className="ui-caption start-scope-detail">
                {family.preferred.inventory.standardSelection === undefined
                  ? `${presentCheckType(family.preferred).label} · ${summary.detail}`
                  : summary.detail}
                {family.preferred.serverCompatible
                  ? ""
                  : " · not supported by this server"}
              </span>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

function ScopeOperations({
  operations,
  identity,
}: {
  operations: { method: string; path: string }[];
  identity: string;
}) {
  const [cursor, setCursor] = useState({ identity, page: 0 });
  const page =
    cursor.identity === identity
      ? Math.min(
          cursor.page,
          Math.max(0, Math.ceil(operations.length / 50) - 1),
        )
      : 0;
  const start = page * 50;
  return (
    <details className="start-scope-operations">
      <summary>View endpoints</summary>
      <ul>
        {operations.slice(start, start + 50).map((operation) => (
          <li key={`${operation.method} ${operation.path}`}>
            <span>{operation.method}</span>
            <code>{operation.path}</code>
          </li>
        ))}
      </ul>
      {operations.length <= 50 ? null : (
        <div className="start-scope-pager">
          <Button
            size="xs"
            disabled={page === 0}
            onClick={() => setCursor({ identity, page: page - 1 })}
          >
            Previous endpoints
          </Button>
          <span>
            {start + 1}–{Math.min(start + 50, operations.length)} of{" "}
            {operations.length}
          </span>
          <Button
            size="xs"
            disabled={start + 50 >= operations.length}
            onClick={() => setCursor({ identity, page: page + 1 })}
          >
            Next endpoints
          </Button>
        </div>
      )}
    </details>
  );
}

export function ScopeCard({ model }: { model: StartCheck }) {
  const id = useId();
  const scope = model.materialScope;
  if (!scope.show) return null;
  const source = scope.content.material;
  const sourceHref =
    source === undefined
      ? undefined
      : artifactDetailPath(
          { kind: "project", id: model.projectId },
          source.artifact,
        );
  const engagement =
    scope.engagement.data?.kind === "engagement"
      ? scope.engagement.data.document
      : undefined;
  return (
    <section className="start-card start-scope-card" aria-labelledby={id}>
      <div className="start-card-head">
        <h3 id={id}>Scope</h3>
      </div>
      {!scope.showInventory ? null : (
        <>
          <p className="start-scope-size">{scope.summary}</p>
          {scope.writing ? (
            <p className="ui-caption start-tight">
              From your applied checklist.
            </p>
          ) : source === undefined ? (
            <p className="ui-caption">{scope.base?.detail}</p>
          ) : (
            <>
              <p className="start-scope-source">
                <code>
                  {source.artifact.namespace}/{source.artifact.name}@
                  {source.artifact.revision}
                </code>
              </p>
              {scope.content.loading ? (
                <p className="ui-caption" role="status">
                  Reading the selected material's scope…
                </p>
              ) : null}
              {!scope.content.eligible ? (
                <p className="ui-caption">
                  This material is larger than the{" "}
                  {AUTOMATIC_DOCUMENT_BYTES / 1024} KiB automatic scope limit.
                  Open Preview to inspect its content.
                </p>
              ) : scope.content.data === null ? (
                <p className="ui-caption">
                  The material's scope could not be determined from this
                  document. The Server validates inputs at start.
                </p>
              ) : null}
              {scope.content.error === null ? null : (
                <ErrorNotice
                  error={scope.content.error}
                  context="The selected material's scope could not be read."
                  onRetry={() => void scope.content.refetch()}
                  retryPending={scope.content.isFetching}
                />
              )}
              {scope.content.data?.kind === "api" ? (
                <>
                  <ScopeOperations
                    operations={scope.content.data.operations}
                    identity={`${model.projectId}/${source.artifact.namespace}/${source.artifact.name}@${source.artifact.revision}`}
                  />
                  {scope.content.data.partial ? (
                    <p className="ui-caption">
                      Referenced paths are resolved at start; the list shows
                      inline endpoints only.
                    </p>
                  ) : null}
                </>
              ) : null}
              <Link to={sourceHref!} className="start-scope-preview">
                Preview material →
              </Link>
            </>
          )}
          <p className="ui-caption">{scope.base?.detail}</p>
        </>
      )}
      {scope.engagementSlots.length === 0 ? null : (
        <div className="start-engagement">
          <h4>Engagement</h4>
          <ul className="start-material-list">
            {scope.engagementSlots.map((slot) => (
              <MaterialSlot
                key={slot.name}
                slot={slot}
                complete={model.materials.complete}
                addHref={projectPaths(model.projectId).addMaterial}
                onChoose={model.chooseMaterial}
              />
            ))}
          </ul>
          {scope.engagement.loading ? (
            <p className="ui-caption" role="status">
              Reading engagement scope…
            </p>
          ) : null}
          {engagement === undefined ? null : (
            <Facts
              items={[
                { term: "Target", value: engagement.target.origins.join(", ") },
                {
                  term: "Allowed",
                  value: engagement.scope.routes
                    .map(
                      (route) =>
                        `${route.pathPrefixes.map((prefix) => route.origin + prefix).join(", ")} · ${route.methods.join(", ")}`,
                    )
                    .join("; "),
                },
                {
                  term: "Excluded",
                  value:
                    engagement.scope.exclusions.length === 0
                      ? "None"
                      : engagement.scope.exclusions
                          .map(
                            (route) =>
                              `${route.pathPrefixes.map((prefix) => route.origin + prefix).join(", ")} · ${route.methods.join(", ")}`,
                          )
                          .join("; "),
                },
                {
                  term: "Access",
                  value:
                    engagement.accessMode === "blackbox"
                      ? "Blackbox"
                      : "Source-assisted",
                },
                {
                  term: "Identities",
                  value: engagement.identities
                    .map((identity) =>
                      [identity.role ?? identity.kind, identity.tenant]
                        .filter(Boolean)
                        .join(" · "),
                    )
                    .join("; "),
                },
              ]}
            />
          )}
          {engagement === undefined ? null : (
            <details className="start-engagement-limits">
              <summary>Limits and technical details</summary>
              <Facts
                items={[
                  {
                    term: "Active seconds",
                    value: engagement.budgets.activeSeconds,
                  },
                  { term: "Rounds", value: engagement.budgets.rounds },
                  {
                    term: "Child runs",
                    value: engagement.budgets.submittedRuns,
                  },
                  {
                    term: "Evidence bytes",
                    value: engagement.budgets.evidenceBytes,
                  },
                  {
                    term: "Addresses",
                    value: engagement.scope.routes
                      .flatMap((route) => route.addressRanges)
                      .join(", "),
                  },
                ]}
              />
            </details>
          )}
          {scope.engagement.error === null ? null : (
            <ErrorNotice
              error={scope.engagement.error}
              context="Engagement scope could not be read."
              onRetry={() => void scope.engagement.refetch()}
              retryPending={scope.engagement.isFetching}
            />
          )}
          {scope.engagement.material !== undefined &&
          engagement === undefined &&
          !scope.engagement.loading ? (
            <p className="ui-caption">
              Inspect the saved Engagement in{" "}
              <Link
                to={artifactDetailPath(
                  { kind: "project", id: model.projectId },
                  scope.engagement.material.artifact,
                )}
              >
                Material Preview
              </Link>
              .
            </p>
          ) : null}
          {scope.engagement.material === undefined ? null : (
            <Link
              className="start-scope-preview"
              to={artifactDetailPath(
                { kind: "project", id: model.projectId },
                scope.engagement.material.artifact,
              )}
            >
              Preview engagement →
            </Link>
          )}
          <p className="ui-caption">
            The Server validates the exact engagement, permissions and
            compatibility at start.
          </p>
        </div>
      )}
    </section>
  );
}
