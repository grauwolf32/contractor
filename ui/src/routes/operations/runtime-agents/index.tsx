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
        {bindings.error === null ? null : (
          <ErrorNotice error={bindings.error} />
        )}
        {principals.error !== null ? (
          <ErrorNotice error={principals.error} />
        ) : principals.isPending ? (
          <p className="loading-copy">Loading Runtime Agents…</p>
        ) : principals.data.items.length === 0 ? (
          <p className="compact-empty">
            No Runtime Agent certificate principal has registered yet.
          </p>
        ) : (
          <div className="runtime-principal-grid">
            {principals.data.items.map((principal) => (
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
