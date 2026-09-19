import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";

import { MobileSectionPicker } from "../../app/mobile-section-picker";
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
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
        });
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
  const queryClient = useQueryClient();
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
  const refresh = () => {
    void query.refetch();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
    });
  };

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
          <h2>Operations</h2>
          <p className="lede">
            Monitor Runtime Agents, inspect allocations and manage execution
            settings.
          </p>
        </div>
        {independentRead || personalSettings ? null : (
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={refresh}
          >
            {query.isFetching ? "Refreshing…" : "Refresh snapshot"}
          </button>
        )}
      </header>

      <nav
        className="operations-navigation section-navigation"
        aria-label="Operations sections"
      >
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
      <MobileSectionPicker
        label="Operations section"
        value={
          [...navigation]
            .reverse()
            .find(
              (item) =>
                normalizedPath === item.to ||
                normalizedPath.startsWith(item.to + "/"),
            )?.to ?? "/operations"
        }
        options={navigation.filter(
          (item) => authorized || item.to === "/operations/settings",
        )}
        state={location.state}
      />

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
          <Outlet
            context={
              {
                snapshot: query.data,
                refresh,
                refreshing: query.isFetching,
              } satisfies OperationsOutletContext
            }
          />
          <details className="panel operations-snapshot-record">
            <summary>Diagnostics: snapshot and live connection</summary>
            <dl className="key-value-list">
              <div>
                <dt>Generation</dt>
                <dd>
                  <code>{query.data.cursor.generation}</code>
                </dd>
              </div>
              <div>
                <dt>Snapshot revision</dt>
                <dd>
                  <code>{query.data.cursor.revision}</code>
                </dd>
              </div>
            </dl>
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
            <p className="muted-copy">
              These are transport diagnostics. Runtime readiness and Run wait
              reasons are separate Server-owned facts. Refresh the snapshot if
              live updates are unavailable.
            </p>
          </details>
        </>
      )}
    </section>
  );
}
