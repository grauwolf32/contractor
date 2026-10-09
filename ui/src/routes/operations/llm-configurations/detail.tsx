import { useQuery } from "@tanstack/react-query";
import { type RefObject, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  getConfiguration,
  isConfigurationKind,
  type ConfigurationResource,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
} from "../../../api/workflows";
import { compactDigest } from "../../../app/format";
import { QueryView } from "../../../app/query-view";
import { ErrorNotice } from "../../../app/error-notice";
import { DetailHeader, IdChip, TechnicalDetails } from "../../../ui";
import { DisclosureChevron, Glance, ScopeChip } from "../common";
import { ConfigurationBodyView } from "./body";
import { LLMGatewayPublicationForm, ModelPolicyPublicationForm } from "./forms";
import { configurationListPath, KIND_LABELS } from "./kinds";

function writable(resource: ConfigurationResource): boolean {
  return (
    resource.ref.kind === "model-policies" ||
    resource.ref.kind === "llm-gateways"
  );
}

/** Opens the clone draft, scrolls to it and focuses its first field. */
function openDraft(clone: RefObject<HTMLDetailsElement | null>) {
  const details = clone.current;
  if (details === null) return;
  details.open = true;
  details.scrollIntoView?.({ block: "start" });
  details.querySelector("input")?.focus();
}

function LoadedConfiguration({
  resource,
  clone,
}: {
  resource: ConfigurationResource;
  clone: RefObject<HTMLDetailsElement | null>;
}) {
  const [published, setPublished] = useState<ConfigurationResource>();
  return (
    <>
      {published === undefined ? null : (
        <div className="notice notice-success" role="status">
          <strong>
            Published {published.ref.name}@{published.ref.version}
          </strong>
          <span>
            Server assigned digest <code>{published.ref.digest}</code>.
          </span>
          <Link
            to={`/operations/configurations/${published.ref.kind}/${encodeURIComponent(published.ref.name)}/${encodeURIComponent(published.ref.version)}`}
          >
            Open the new version
          </Link>
        </div>
      )}
      <section
        className="ops-panel"
        aria-labelledby="configuration-values-heading"
      >
        <div className="ops-panel-head">
          <h3 id="configuration-values-heading" className="ops-panel-title">
            Published values
          </h3>
          <span className="ops-panel-aside">
            The existing version and digest are never edited.
          </span>
        </div>
        <ConfigurationBodyView resource={resource} />
      </section>
      <TechnicalDetails
        summary="Published identity"
        description={`Kind, source and digest · ${resource.source}`}
      >
        <Glance
          items={[
            ["Kind", <code key="kind">{resource.ref.kind}</code>],
            ["Source", resource.source],
            [
              "Digest",
              <IdChip
                key="digest"
                value={resource.ref.digest}
                display={compactDigest(resource.ref.digest)}
                label="configuration digest"
              />,
            ],
          ]}
        />
      </TechnicalDetails>
      {writable(resource) ? (
        <details className="ops-disclosure configuration-clone" ref={clone}>
          <summary>
            <DisclosureChevron />
            New version draft
          </summary>
          {resource.ref.kind === "model-policies" ? (
            <ModelPolicyPublicationForm
              source={resource}
              onPublished={setPublished}
            />
          ) : (
            <LLMGatewayPublicationForm
              source={resource}
              onPublished={setPublished}
            />
          )}
        </details>
      ) : (
        <p className="ops-note">
          This configuration kind is operator-authored and read-only. Publish
          new versions through the configuration source; this page shows the
          recorded values.
        </p>
      )}
    </>
  );
}

export function ConfigurationDetailRoute() {
  const api = usePublicAPI();
  const clone = useRef<HTMLDetailsElement>(null);
  const { kind = "", name = "", version = "" } = useParams();
  const configurationKind = isConfigurationKind(kind) ? kind : undefined;
  const valid =
    configurationKind !== undefined &&
    CONFIG_NAME_PATTERN.test(name) &&
    CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.configurations.detail(kind, name, version),
    queryFn: () => {
      if (configurationKind === undefined) {
        throw new TypeError("Configuration kind is invalid");
      }
      return getConfiguration(api, configurationKind, name, version);
    },
    enabled: valid,
  });
  const resource = query.data;
  return (
    <div className="ops-stack ops-detail">
      <DetailHeader
        breadcrumb={[
          { label: "LLM configurations", to: "/operations/configurations" },
          ...(configurationKind === undefined
            ? []
            : [
                {
                  label: KIND_LABELS[configurationKind].plural,
                  to: configurationListPath(configurationKind),
                },
              ]),
        ]}
        title={`${name}@${version}`}
        status={
          resource === undefined ? undefined : (
            <ScopeChip>{resource.source}</ScopeChip>
          )
        }
        actions={
          resource !== undefined && writable(resource) ? (
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              onClick={() => openDraft(clone)}
            >
              Clone to new version
            </button>
          ) : undefined
        }
      />
      {!valid ? (
        <ErrorNotice error={new Error("Configuration route is invalid")} />
      ) : (
        <QueryView
          query={query}
          loading={
            <p className="ops-loading" role="status">
              Loading configuration…
            </p>
          }
          onRetry={() => void query.refetch()}
        >
          {(loaded) => (
            <LoadedConfiguration
              key={loaded.ref.digest}
              resource={loaded}
              clone={clone}
            />
          )}
        </QueryView>
      )}
    </div>
  );
}
