import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";

import { usePublicAPI } from "../../api/context";
import {
  getOperationsSnapshot,
  type OperationsSnapshot,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { useSession } from "../../auth/session";
import { useRunEvents } from "../../events/context";
import type {
  OperationsEventData,
  RunEventConnectionState,
  RunResyncReason,
} from "../../events/run-events";
import { ErrorNotice } from "../artifacts/common";
import type { OperationsOutletContext } from "./context";

const navigation = [
  { to: "/operations", label: "Overview", end: true },
  { to: "/operations/runtime-agents", label: "Runtime Agents" },
  { to: "/operations/allocations", label: "Allocations" },
  { to: "/operations/performance", label: "Performance" },
  { to: "/operations/configurations", label: "LLM configurations" },
  { to: "/operations/credentials", label: "Credentials" },
  { to: "/operations/settings", label: "Settings" },
] as const;

function OperationsLiveSubscription({
  snapshot,
  onConnection,
  onError,
  onResync,
}: {
  snapshot: OperationsSnapshot;
  onConnection: (state: RunEventConnectionState) => void;
  onError: (message?: string) => void;
  onResync: (reason: RunResyncReason) => void;
}) {
  const events = useRunEvents();
  const queryClient = useQueryClient();

  useEffect(() => {
    const invalidateSnapshot = () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.snapshot,
      });
    const invalidateSchedulerSettings = () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.schedulerSettings,
      });
    const changed = (resource: OperationsEventData["resource"]) => {
      if (resource === "configuration") {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.configurations.all,
        });
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeConfigs.all,
        });
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeLabels.all,
        });
      }
      if (resource === "credential") {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.credentials.all,
        });
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        });
      }
      if (resource === "runtimeAgent") {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
        });
      }
      if (resource === "schedulerSettings") {
        void invalidateSchedulerSettings();
      }
      void invalidateSnapshot();
    };
    const subscription = events.subscribeOperations(snapshot.cursor, {
      onOperationsEvent: (event) => changed(event.data.resource),
      onResync: (reason) => {
        onResync(reason);
        void invalidateSnapshot();
        void invalidateSchedulerSettings();
      },
      onStateChange: (state) => {
        onConnection(state);
        if (state === "live") {
          void invalidateSchedulerSettings();
        }
      },
      onError: (message) => onError(message),
    });
    return () => subscription.unsubscribe();
  }, [events, onConnection, onError, onResync, queryClient, snapshot]);
  return null;
}

export function OperationsLayoutRoute() {
  const api = usePublicAPI();
  const location = useLocation();
  const { session } = useSession();
  const authorized =
    session?.principal.capabilities.includes("operations") === true;
  const normalizedPath = location.pathname.replace(/\/+$/, "");
  const personalSettings =
    !authorized && normalizedPath === "/operations/settings";
  const independentRead =
    normalizedPath === "/operations/performance" ||
    normalizedPath === "/operations/allocations/completed";
  const [connection, setConnection] =
    useState<RunEventConnectionState>("connecting");
  const [liveError, setLiveError] = useState<string>();
  const [resyncReason, setResyncReason] = useState<RunResyncReason>();
  const recordConnection = useCallback((state: RunEventConnectionState) => {
    setConnection(state);
    if (state === "live") {
      setLiveError(undefined);
    }
  }, []);
  const recordLiveError = useCallback((message?: string) => {
    setLiveError(message);
  }, []);
  const recordResync = useCallback((reason: RunResyncReason) => {
    setResyncReason(reason);
    setLiveError(undefined);
  }, []);
  const query = useQuery({
    queryKey: queryKeys.operations.snapshot,
    queryFn: () => getOperationsSnapshot(api),
    enabled: authorized && !independentRead,
  });

  if (!authorized && !personalSettings) {
    return (
      <section className="route-page operations-page">
        <p className="eyebrow">Operations capability required</p>
        <h2>Operations</h2>
        <div className="notice notice-error" role="alert">
          This session is not authorized to observe or manage Operations
          resources.
        </div>
      </section>
    );
  }

  const liveKey =
    query.data === undefined
      ? "none"
      : `${query.data.cursor.generation}:${query.data.cursor.revision}:${query.dataUpdatedAt}`;
  return (
    <section className="route-page operations-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Control Plane observation and configuration</p>
          <h2>Operations</h2>
          <p className="lede">
            REST snapshots are authoritative. Live events only invalidate this
            view; they never predict allocation or configuration state.
          </p>
        </div>
        {independentRead || personalSettings ? null : (
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh snapshot"}
          </button>
        )}
      </header>

      <nav className="operations-navigation" aria-label="Operations sections">
        {navigation
          .filter((item) => authorized || item.to === "/operations/settings")
          .map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={"end" in item ? item.end : false}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {item.label}
            </NavLink>
          ))}
      </nav>

      {independentRead || personalSettings ? (
        <Outlet />
      ) : query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading authoritative Operations snapshot…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <OperationsLiveSubscription
            key={liveKey}
            snapshot={query.data}
            onConnection={recordConnection}
            onError={recordLiveError}
            onResync={recordResync}
          />
          <div className={`live-status live-${connection}`} role="status">
            <span className="status-dot" aria-hidden="true" />
            Operations events: {connection}
            {resyncReason === undefined
              ? null
              : ` · REST resync after ${resyncReason.replaceAll("_", " ")}`}
          </div>
          {liveError === undefined ? null : (
            <div className="notice notice-warning" role="alert">
              <strong>{liveError}</strong>
              <p>
                Manual snapshot refresh remains available and authoritative.
              </p>
            </div>
          )}
          <Outlet
            context={
              {
                snapshot: query.data,
                refresh: () => void query.refetch(),
                refreshing: query.isFetching,
              } satisfies OperationsOutletContext
            }
          />
        </>
      )}
    </section>
  );
}
