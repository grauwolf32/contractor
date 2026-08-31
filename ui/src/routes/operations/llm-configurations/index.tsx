import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  CONFIGURATION_KINDS,
  listConfigurations,
  type ConfigurationKind,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls, ErrorNotice } from "../../artifacts/common";

const kindLabels: Record<ConfigurationKind, string> = {
  "agent-templates": "AgentTemplates (read-only)",
  "execution-configs": "ExecutionConfigs (read-only)",
  "model-policies": "ModelPolicies",
  "llm-gateways": "LLMGatewayConfigs",
};

export function ConfigurationListRoute() {
  const api = usePublicAPI();
  const [kind, setKind] = useState<ConfigurationKind>("model-policies");
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.configurations.list(kind, cursor),
    queryFn: () =>
      listConfigurations(api, kind, cursor === undefined ? {} : { cursor }),
  });
  return (
    <div className="panel operations-library">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Immutable published configuration</p>
          <h3>LLM configurations</h3>
          <p className="muted-copy">
            All kinds are inspectable. Only ModelPolicy and LLMGatewayConfig can
            be cloned into a new create-only managed version.
          </p>
        </div>
        <label className="compact-select">
          Kind
          <select
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as ConfigurationKind);
              setCursors([undefined]);
            }}
          >
            {CONFIGURATION_KINDS.map((candidate) => (
              <option key={candidate} value={candidate}>
                {kindLabels[candidate]}
              </option>
            ))}
          </select>
        </label>
      </div>
      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading configurations…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          No published {kindLabels[kind]} versions.
        </div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Exact version</th>
                <th>Source</th>
                <th>Digest</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((resource) => (
                <tr
                  key={`${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`}
                >
                  <td>
                    <strong>
                      {resource.ref.name}@{resource.ref.version}
                    </strong>
                  </td>
                  <td>
                    <span className="state-badge">{resource.source}</span>
                  </td>
                  <td>
                    <code title={resource.ref.digest}>
                      {resource.ref.digest.slice(0, 22)}…
                    </code>
                  </td>
                  <td>
                    <Link
                      to={`/operations/configurations/${resource.ref.kind}/${encodeURIComponent(resource.ref.name)}/${encodeURIComponent(resource.ref.version)}`}
                    >
                      {resource.ref.kind === "model-policies" ||
                      resource.ref.kind === "llm-gateways"
                        ? "Inspect / clone"
                        : "Inspect"}
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Configuration pages"
        canGoBack={cursors.length > 1}
        {...(query.data?.page.hasMore === true &&
        query.data.page.nextCursor !== undefined
          ? { nextCursor: query.data.page.nextCursor }
          : {})}
        onBack={() =>
          setCursors((current) =>
            current.slice(0, Math.max(1, current.length - 1)),
          )
        }
        onNext={(next) => setCursors((current) => [...current, next])}
      />
    </div>
  );
}
