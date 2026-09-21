import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  listRuntimeAgentPrincipals,
  listRuntimeLabels,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../artifacts/common";
import { useOperationsSnapshot } from "../context";
import { AgentCard } from "./agent-card";
import { compareByName, isOfflineIdentity } from "./identity";
import { OfflineIdentities } from "./offline-identities";
import { ProcessInventory } from "./process-inventory";
import "./styles.css";

export function RuntimeAgentListRoute() {
  const api = usePublicAPI();
  const [connection, setConnection] = useState("all");
  const { snapshot } = useOperationsSnapshot();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(timer);
  }, []);
  const principals = useQuery({
    queryKey: queryKeys.operations.runtimeAgentPrincipals.list(),
    queryFn: () => listRuntimeAgentPrincipals(api),
  });
  const bindings = useQuery({
    queryKey: queryKeys.operations.runtimeLabels.list(),
    queryFn: () => listRuntimeLabels(api),
  });
  const items = principals.data?.items ?? [];
  const online =
    connection === "offline"
      ? []
      : items
          .filter((principal) => !isOfflineIdentity(principal))
          .sort(compareByName);
  const offline =
    connection === "online"
      ? []
      : items.filter(isOfflineIdentity).sort(compareByName);
  return (
    <>
      <div className="operations-library runtime-agent-library">
        <div className="section-heading">
          <div>
            <h3>Runtime Agents</h3>
            <p className="muted-copy">
              Availability, current work and saved Agent labels.
            </p>
          </div>
          <span>
            {snapshot.runtimeAgents.length} online ·{" "}
            {principals.data?.items.length ?? 0} identities loaded
          </span>
        </div>
        <label className="compact-select">
          Connection
          <select
            aria-label="Agent connection"
            value={connection}
            onChange={(event) => setConnection(event.target.value)}
          >
            <option value="all">All agents · online first</option>
            <option value="online">Online</option>
            <option value="offline">Offline</option>
          </select>
        </label>
        {bindings.error === null ? null : (
          <ErrorNotice error={bindings.error} />
        )}
        {principals.error !== null ? (
          <ErrorNotice error={principals.error} />
        ) : principals.isPending ? (
          <p className="loading-copy" role="status">
            Loading Runtime Agents…
          </p>
        ) : online.length === 0 && offline.length === 0 ? (
          <p className="compact-empty">
            No agents match this connection filter.
          </p>
        ) : (
          <>
            {online.length === 0 ? null : (
              <div className="runtime-principal-grid">
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
        )}
        <p className="runtime-agent-availability-note">
          Availability is reported by Server. Each Workflow still requires
          compatible capabilities.
        </p>
      </div>
      <ProcessInventory />
    </>
  );
}
