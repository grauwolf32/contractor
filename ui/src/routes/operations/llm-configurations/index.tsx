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
import { ButtonLink, FilterChips } from "../../../ui";
import { EntryHeading, OpsSection, ScopeChip } from "../common";
import { configurationVersionPath, KIND_LABELS, KIND_ORDER } from "./kinds";

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
        isEmpty={(data) => data.items.length === 0}
        empty={
          <p className="ops-empty">
            No published {label.plural.toLowerCase()} versions.
          </p>
        }
      >
        {(data) => (
          <div className="ops-resource-list">
            {data.items.map((resource) => (
              <details
                className="ops-resource-row ops-resource-disclosure"
                key={`${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`}
              >
                <summary className="ops-resource-heading">
                  <EntryHeading
                    icon="docs"
                    title={
                      <strong>
                        {resource.ref.name}@{resource.ref.version}
                      </strong>
                    }
                    subtitle={label.singular}
                  >
                    <ScopeChip>{resource.source}</ScopeChip>
                  </EntryHeading>
                </summary>
                <div className="ops-resource-body">
                  <code className="ops-digest" title={resource.ref.digest}>
                    {compactDigest(resource.ref.digest)}
                  </code>
                  <ButtonLink
                    size="sm"
                    to={configurationVersionPath(resource.ref)}
                  >
                    {resource.ref.kind === "agent-templates"
                      ? "Inspect"
                      : "Inspect / clone"}
                  </ButtonLink>
                </div>
              </details>
            ))}
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
            <p>Agent templates are read-only here.</p>
          </>
        }
        eyebrow="Server-wide"
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
        <p className="ops-inventory-note">
          Inspect an exact version. Publishing a clone does not select it for
          execution.
        </p>
        {kind === "agent-templates" ? (
          <p className="ui-caption">
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
