import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  listAllRuntimeAgentPrincipals,
  listAllRuntimeLabels,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../../app/error-notice";
import { FilterChips } from "../../../ui";
import { OpsSection } from "../common";
import { useOperationsSnapshot } from "../context";
import { AgentCard } from "./agent-card";
import { compareByName, isOfflineIdentity } from "./identity";
import { OfflineIdentities } from "./offline-identities";
import { ProcessInventory } from "./process-inventory";
import { QueryView } from "../../../app/query-view";

type Connection = "all" | "online" | "offline";

export function RuntimeAgentListRoute() {
  const api = usePublicAPI();
  const [connection, setConnection] = useState<Connection>("all");
  const { snapshot } = useOperationsSnapshot();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(timer);
  }, []);
  const principals = useQuery({
    queryKey: queryKeys.operations.runtimeAgentPrincipals.list(),
    queryFn: () => listAllRuntimeAgentPrincipals(api),
  });
  const bindings = useQuery({
    queryKey: queryKeys.operations.runtimeLabels.list(),
    queryFn: () => listAllRuntimeLabels(api),
  });
  const items = principals.data?.items ?? [];
  const allOnline = items
    .filter((principal) => !isOfflineIdentity(principal))
    .sort(compareByName);
  const allOffline = items.filter(isOfflineIdentity).sort(compareByName);
  const online = connection === "offline" ? [] : allOnline;
  const offline = connection === "online" ? [] : allOffline;
  return (
    <div className="ops-stack">
      <OpsSection
        id="runtime-agents-heading"
        title="Runtime Agents"
        description="Availability, current work and saved Agent labels."
        aside={`${snapshot.runtimeAgents.length} online · ${principals.data?.items.length ?? 0} identities loaded`}
      >
        {principals.data === undefined ? null : (
          <FilterChips<Connection>
            label="Agent connection"
            value={connection}
            onChange={setConnection}
            options={[
              { value: "all", label: "All", count: items.length },
              { value: "online", label: "Online", count: allOnline.length },
              { value: "offline", label: "Offline", count: allOffline.length },
            ]}
          />
        )}
        {bindings.error === null ? null : (
          <ErrorNotice error={bindings.error} />
        )}
        <QueryView
          query={principals}
          loading={
            <p className="ops-loading" role="status">
              Loading Runtime Agents…
            </p>
          }
          onRetry={() => void principals.refetch()}
        >
          {() =>
            online.length === 0 && offline.length === 0 ? (
              <p className="ops-empty">
                No agents match this connection filter.
              </p>
            ) : (
              <>
                {online.length === 0 ? null : (
                  <div className="ops-agent-grid">
                    {online.map((principal) => (
                      <AgentCard
                        key={principal.runtimeAgentId}
                        principal={principal}
                        bindings={bindings.data?.items}
                        allocations={snapshot.allocations}
                        now={now}
                      />
                    ))}
                  </div>
                )}
                <OfflineIdentities
                  principals={offline}
                  bindings={bindings.data?.items}
                  now={now}
                  open={connection === "offline"}
                />
              </>
            )
          }
        </QueryView>
        <p className="ops-note">
          Availability is reported by Server. Each Workflow still requires
          compatible capabilities.
        </p>
      </OpsSection>
      <ProcessInventory />
    </div>
  );
}
