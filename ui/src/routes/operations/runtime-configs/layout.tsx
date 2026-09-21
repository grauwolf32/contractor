import { Outlet } from "react-router";

/**
 * Runtime configuration hub under Operations. Live invalidation and the
 * Refresh control come from the Operations layout, which already subscribes
 * to configuration and credential events for every section.
 */
export function RuntimeConfigurationLayout() {
  return (
    <>
      <div className="section-heading configuration-scope-heading">
        <div>
          <p className="eyebrow">Execution defaults</p>
          <h3>Runtime configuration</h3>
          <p className="muted-copy">
            RuntimeConfig versions, label bindings and Runtime service
            credentials apply to every Run on this server.
          </p>
        </div>
        <span className="state-badge">Server-wide</span>
      </div>
      <Outlet />
    </>
  );
}
