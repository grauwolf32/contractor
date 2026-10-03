import "../configuration-reading.css";
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
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { QueryView } from "../../../app/query-view";

const kindLabels: Record<ConfigurationKind, string> = {
  "agent-templates": "AgentTemplates (read-only)",
  "execution-configs": "ExecutionConfigs (read-only)",
  "model-policies": "ModelPolicies",
  "llm-gateways": "LLMGatewayConfigs",
};

export function ConfigurationListRoute() {
  const api = usePublicAPI();
  const [kind, setKind] = useState<ConfigurationKind>("model-policies");
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.configurations.list(kind, cursor),
    queryFn: () =>
      listConfigurations(api, kind, cursor === undefined ? {} : { cursor }),
  });
  return (
    <div className="panel operations-library">
      <div className="section-heading">
        <div>
          <h3>LLM configurations</h3>
          <span className="state-badge">Server-wide</span>
          <p className="muted-copy">
            Model policies define LLM behavior and limits; gateways route model
            requests. New versions take effect when selected for future
            executions.
          </p>
        </div>
        <label className="compact-select">
          Kind
          <select
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as ConfigurationKind);
              pages.reset();
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
      {kind === "agent-templates" ? (
        <p>
          <Link to="/catalog/agents">
            Explore Agent instructions, Skills and Workflow usage in Catalog →
          </Link>
        </p>
      ) : null}
      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            Loading configurations…
          </p>
        }
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="compact-empty">
            No published {kindLabels[kind]} versions.
          </div>
        }
      >
        {(data) => (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Source</th>
                  <th>Digest</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((resource) => (
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
      </QueryView>
      <CursorControls
        label="Configuration pages"
        {...pages.controls(query.data?.page)}
      />
    </div>
  );
}
