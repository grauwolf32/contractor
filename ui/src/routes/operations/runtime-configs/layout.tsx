import { Outlet } from "react-router";

import { ScopeChip } from "../common";

/**
 * Runtime configuration hub under Operations. Live invalidation and the
 * Refresh control come from the Operations layout, which already subscribes
 * to configuration and credential events for every section.
 */
export function RuntimeConfigurationLayout() {
  return (
    <div className="ops-stack">
      <header className="ops-section-head">
        <div className="ops-section-heading">
          <p className="ops-eyebrow">Execution defaults</p>
          <h2 className="ops-section-title">Runtime configuration</h2>
          <p className="ops-section-description">
            RuntimeConfig versions, label bindings and Runtime service
            credentials apply to every Run on this server.
          </p>
        </div>
        <ScopeChip>Server-wide</ScopeChip>
      </header>
      <Outlet />
    </div>
  );
}
