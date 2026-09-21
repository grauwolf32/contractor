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
  const visible = [...(principals.data?.items ?? [])]
    .filter(
      (principal) =>
        connection === "all" ||
        (connection === "online"
          ? principal.live !== undefined
          : principal.live === undefined),
    )
    .sort(
      (a, b) =>
        Number(b.live !== undefined) - Number(a.live !== undefined) ||
        a.runtimeAgentId.localeCompare(b.runtimeAgentId),
    );
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
        ) : visible.length === 0 ? (
          <p className="compact-empty">
            No agents match this connection filter.
          </p>
        ) : (
          <div className="runtime-principal-grid">
            {visible.map((principal) => (
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
        <p className="runtime-agent-availability-note">
          Availability is reported by Server. Each Workflow still requires
          compatible capabilities.
        </p>
      </div>
      <ProcessInventory />
    </>
  );
}
