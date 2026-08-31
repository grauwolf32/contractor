import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  getConfiguration,
  isConfigurationKind,
  type ConfigurationResource,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
} from "../../../api/workflows";
import { ErrorNotice } from "../../artifacts/common";
import { ConfigurationBodyView } from "./body";
import { LLMGatewayPublicationForm, ModelPolicyPublicationForm } from "./forms";

function LoadedConfiguration({
  resource,
}: {
  resource: ConfigurationResource;
}) {
  const [published, setPublished] = useState<ConfigurationResource>();
  return (
    <>
      <div className="metadata-grid panel">
        <div>
          <dt>Kind</dt>
          <dd>{resource.ref.kind}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>
            <span className="state-badge">{resource.source}</span>
          </dd>
        </div>
        <div>
          <dt>Immutable digest</dt>
          <dd>
            <code>{resource.ref.digest}</code>
          </dd>
        </div>
      </div>
      <div className="panel configuration-inspector">
        <p className="eyebrow">Safe typed body</p>
        <h3>Published values</h3>
        <ConfigurationBodyView resource={resource} />
      </div>
      {published === undefined ? null : (
        <div className="notice notice-success" role="status">
          <strong>
            Published {published.ref.name}@{published.ref.version}
          </strong>
          <span>
            Server assigned exact digest <code>{published.ref.digest}</code>.
          </span>
          <Link
            to={`/operations/configurations/${published.ref.kind}/${encodeURIComponent(published.ref.name)}/${encodeURIComponent(published.ref.version)}`}
          >
            Open the new immutable version
          </Link>
        </div>
      )}
      {resource.ref.kind === "model-policies" ? (
        <ModelPolicyPublicationForm
          source={resource}
          onPublished={setPublished}
        />
      ) : resource.ref.kind === "llm-gateways" ? (
        <LLMGatewayPublicationForm
          source={resource}
          onPublished={setPublished}
        />
      ) : (
        <div className="notice notice-warning">
          <strong>
            This configuration kind is operator-authored and read-only.
          </strong>
          <p>
            Its first UI editor is intentionally deferred; this exact version
            remains inspectable.
          </p>
        </div>
      )}
    </>
  );
}

export function ConfigurationDetailRoute() {
  const api = usePublicAPI();
  const { kind = "", name = "", version = "" } = useParams();
  const configurationKind = isConfigurationKind(kind) ? kind : undefined;
  const valid =
    configurationKind !== undefined &&
    CONFIG_ID_PATTERN.test(name) &&
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
  return (
    <div className="configuration-detail">
      <Link className="back-link" to="/operations/configurations">
        ← All configurations
      </Link>
      <p className="eyebrow">Exact published configuration</p>
      <h3>
        {name}@{version}
      </h3>
      {!valid ? (
        <ErrorNotice error={new Error("Configuration route is invalid")} />
      ) : query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading exact configuration…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <LoadedConfiguration
          key={query.data.ref.digest}
          resource={query.data}
        />
      )}
    </div>
  );
}
