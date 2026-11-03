import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { listCredentials } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { RUNTIME_CONFIGURATION_PATH } from "../../../app/navigation";
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { EntryHeading, OpsSection } from "../common";
import { SETTINGS_PATH } from "../settings/path";
import { ActiveChip } from "./active-chip";
import { CredentialCreateForm } from "./form";
import { Button } from "../../../ui";
import { RecordedTime } from "../../../app/recorded-time";
import { QueryView } from "../../../app/query-view";

export function CredentialListRoute() {
  const [creating, setCreating] = useState(false);
  const api = usePublicAPI();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.credentials.list(cursor),
    queryFn: () => listCredentials(api, cursor === undefined ? {} : { cursor }),
  });
  return (
    <div className="ops-stack">
      <OpsSection
        id="managed-credentials-heading"
        eyebrow="LLM gateway access"
        title="Managed LLM credentials"
        description={
          <>
            <p>
              Gateway credentials created in Contractor, with their budgets and
              usage. Development tokens configured outside the UI are not listed
              here.
            </p>
            <p>
              <Link to={RUNTIME_CONFIGURATION_PATH}>
                Runtime service credentials
              </Link>
              {" are managed under Configuration; "}
              <Link to={SETTINGS_PATH}>Git SSH keys</Link>
              {" are in Settings."}
            </p>
          </>
        }
        actions={
          <Button
            variant="primary"
            onClick={() => setCreating((value) => !value)}
            aria-expanded={creating}
            aria-controls="ops-create-credential"
          >
            Create LLM credential
          </Button>
        }
      >
        <div id="ops-create-credential" hidden={!creating}>
          <div className="ops-resource-body">
            <CredentialCreateForm />
          </div>
        </div>
        <QueryView
          query={query}
          loading={
            <p className="ops-loading" role="status">
              Loading credentials…
            </p>
          }
          isEmpty={(data) => data.items.length === 0}
          empty={
            <div className="ops-empty">
              <strong>No managed LLM credentials</strong>
              <p>
                Create an LLM credential to manage gateway access and budgets
                here.
              </p>
            </div>
          }
        >
          {(data) => (
            <div className="ops-resource-list">
              {data.items.map((credential) => (
                <details
                  className="ops-resource-row ops-resource-disclosure"
                  key={credential.credentialId}
                >
                  <summary className="ops-resource-heading">
                    <EntryHeading
                      icon="target"
                      title={
                        <strong>
                          {credential.label ?? credential.credentialId}
                        </strong>
                      }
                      subtitle={credential.credentialId}
                    >
                      <ActiveChip />
                    </EntryHeading>
                  </summary>
                  <div className="ops-resource-body">
                    <div className="ops-resource-meta">
                      <span>
                        {credential.llmGateway.gatewayId}@
                        {credential.llmGateway.version}
                      </span>
                      <span>
                        {credential.effectivePolicy.modelPolicies.length} model
                        policies
                      </span>
                      <span>
                        Spend: {credential.consumption?.spend ?? "not observed"}
                      </span>
                      <RecordedTime value={credential.createdAt} />
                    </div>
                    <Link
                      className="ops-details-link"
                      to={`/operations/credentials/${encodeURIComponent(credential.credentialId)}`}
                    >
                      Inspect credential →
                    </Link>
                  </div>
                </details>
              ))}
            </div>
          )}
        </QueryView>
        <CursorControls
          label="Credential pages"
          {...pages.controls(query.data?.page)}
        />
      </OpsSection>
    </div>
  );
}
