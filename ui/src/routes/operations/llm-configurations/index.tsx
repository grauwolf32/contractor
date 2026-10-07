import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  isConfigurationKind,
  listConfigurations,
  type ConfigurationKind,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { QueryView } from "../../../app/query-view";
import { compactDigest } from "../../../app/format";
import { FilterChips } from "../../../ui";
import { OpsSection, ScopeChip } from "../common";
import { KIND_LABELS, KIND_ORDER } from "./kinds";

const DEFAULT_KIND: ConfigurationKind = "model-policies";

function ConfigurationTable({ kind }: { kind: ConfigurationKind }) {
  const api = usePublicAPI();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.configurations.list(kind, cursor),
    queryFn: () =>
      listConfigurations(api, kind, cursor === undefined ? {} : { cursor }),
  });
  const label = KIND_LABELS[kind];
  return (
    <>
      <QueryView
        query={query}
        loading={
          <p className="ops-loading" role="status">
            Loading configurations…
          </p>
        }
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="ops-empty">
            No published {label.plural.toLowerCase()} versions.
          </p>
        }
      >
        {(data) => (
          <div className="ops-table-wrap">
            <table className="ops-table" data-stack="">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Source</th>
                  <th>Digest</th>
                  <th>
                    <span className="ui-visually-hidden">Action</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((resource) => (
                  <tr
                    key={`${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`}
                  >
                    <td data-label="Version">
                      <strong className="ops-mono">
                        {resource.ref.name}@{resource.ref.version}
                      </strong>
                    </td>
                    <td data-label="Source">
                      <ScopeChip>{resource.source}</ScopeChip>
                    </td>
                    <td data-label="Digest">
                      <code className="ops-digest" title={resource.ref.digest}>
                        {compactDigest(resource.ref.digest)}
                      </code>
                    </td>
                    <td data-label="" className="ops-cell-actions">
                      <Link
                        className="ui-btn"
                        data-size="sm"
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
    </>
  );
}

export function ConfigurationListRoute() {
  const [params, setParams] = useSearchParams();
  const requested = params.get("kind");
  const kind =
    requested !== null && isConfigurationKind(requested)
      ? requested
      : DEFAULT_KIND;
  return (
    <div className="ops-stack">
      <OpsSection
        id="llm-configurations-heading"
        title="LLM configurations"
        description={
          <>
            <p>
              Model policies define LLM behavior and limits; gateways route
              model requests. New versions take effect when selected for future
              executions.
            </p>
            <p>Agent templates and execution configs are read-only here.</p>
          </>
        }
        actions={<ScopeChip>Server-wide</ScopeChip>}
      >
        <FilterChips<ConfigurationKind>
          label="Kind"
          value={kind}
          onChange={(next) =>
            setParams(next === DEFAULT_KIND ? {} : { kind: next }, {
              replace: true,
            })
          }
          options={KIND_ORDER.map((candidate) => ({
            value: candidate,
            label: KIND_LABELS[candidate].plural,
          }))}
        />
        {kind === "agent-templates" ? (
          <p className="ops-note">
            <Link to="/catalog/agents">
              Explore Agent instructions, Skills and Workflow usage in the
              Library →
            </Link>
          </p>
        ) : null}
        <ConfigurationTable key={kind} kind={kind} />
      </OpsSection>
    </div>
  );
}
