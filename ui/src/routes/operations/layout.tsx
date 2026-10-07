import "./operations.css";
import { useDocumentTitle } from "../../app/document-title";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router";

import { MobileSectionPicker } from "../../app/mobile-section-picker";
import type { PublicAPI } from "../../api/client";
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
import { QueryView } from "../../app/query-view";
import type { StatusTone } from "../../app/status-tone";
import { StatusChip, StatusGlyph } from "../../ui";
import { Glance, TechnicalDisclosure } from "./common";
import type { OperationsOutletContext } from "./context";
import { RefreshButton } from "../../app/refresh-button";
import { OperationsLiveRefresh } from "./live-refresh";
import { SETTINGS_PATH } from "./settings/path";

const navigation = [
  { to: "/operations", label: "Overview", end: true },
  { to: "/operations/runtime-agents", label: "Runtime Agents" },
  { to: "/operations/allocations", label: "Allocations" },
  { to: "/operations/performance", label: "Performance" },
  { to: "/operations/configuration", label: "Configuration", setup: true },
  {
    to: "/operations/configurations",
    label: "LLM configurations",
    setup: true,
  },
  { to: "/operations/credentials", label: "Credentials", setup: true },
  { to: SETTINGS_PATH, label: "Settings", setup: true },
] as const;

/** First tab of the Setup group; a divider and label precede it. */
const SETUP_GROUP_START = navigation.find((item) => "setup" in item)?.to;

const CONNECTION_TONES: Readonly<Record<RunEventConnectionState, StatusTone>> =
  {
    connecting: "progress",
    live: "done",
    reconnecting: "warning",
    resyncing: "warning",
    error: "blocked",
  };

function snapshotQuery(api: PublicAPI) {
  return {
    queryKey: queryKeys.operations.snapshot,
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      getOperationsSnapshot(api, signal),
  };
}

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
  const api = usePublicAPI();
  const events = useRunEvents();
  const queryClient = useQueryClient();
  const refresher = useRef<OperationsLiveRefresh | null>(null);
  const [initialCursor] = useState(() => snapshot.cursor);

  useEffect(() => {
    const invalidateSchedulerSettings = () =>
      queryClient.invalidateQueries(
        { queryKey: queryKeys.operations.schedulerSettings },
        { cancelRefetch: false },
      );
    const updates = new OperationsLiveRefresh({
      initial: initialCursor,
      // Always reads the Server (joining a read in flight) and rejects when
      // the read fails, so a resync never resumes from the cached cursor.
      refreshSnapshot: async () =>
        (await queryClient.query({ ...snapshotQuery(api), staleTime: 0 }))
          .cursor,
      refreshPrincipals: () =>
        queryClient.invalidateQueries(
          { queryKey: queryKeys.operations.runtimeAgentPrincipals.all },
          { cancelRefetch: false },
        ),
      resume: (cursor) => subscription.resume(cursor),
    });
    refresher.current = updates;
    const changed = (event: {
      cursor: { generation: string; sequence: string };
      data: OperationsEventData;
    }) => {
      const resource = event.data.resource;
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
      if (resource === "schedulerSettings") {
        void invalidateSchedulerSettings();
      }
      updates.event(
        { generation: event.cursor.generation, revision: event.data.revision },
        resource,
      );
    };
    const subscription = events.subscribeOperations(initialCursor, {
      onOperationsEvent: changed,
      onResync: (reason) => {
        onResync(reason);
        updates.resync(reason);
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
    return () => {
      updates.dispose();
      subscription.unsubscribe();
      refresher.current = null;
    };
  }, [
    api,
    events,
    initialCursor,
    onConnection,
    onError,
    onResync,
    queryClient,
  ]);
  useEffect(() => {
    refresher.current?.snapshot(snapshot.cursor);
  }, [snapshot.cursor]);
  return null;
}

/**
 * Transport facts behind a quiet disclosure: the snapshot cursor and the
 * state of the live Operations event stream.
 */
function SnapshotDiagnostics({
  snapshot,
  connection,
  resyncReason,
  liveError,
}: {
  snapshot: OperationsSnapshot;
  connection: RunEventConnectionState;
  resyncReason: RunResyncReason | undefined;
  liveError: string | undefined;
}) {
  return (
    <TechnicalDisclosure
      className="ops-diagnostics operations-snapshot-record"
      summary="Diagnostics: snapshot and live connection"
      description="Snapshot cursor and live update transport, for operators and debugging."
    >
      <div className="ops-diagnostics-body">
        <Glance
          items={[
            ["Generation", <code key="g">{snapshot.cursor.generation}</code>],
            [
              "Snapshot revision",
              <code key="r">{snapshot.cursor.revision}</code>,
            ],
          ]}
        />
        <div className="ops-live-status" role="status">
          <StatusChip tone={CONNECTION_TONES[connection]} size="sm">
            Operations events: {connection}
            {resyncReason === undefined
              ? null
              : ` · REST resync after ${resyncReason.replaceAll("_", " ")}`}
          </StatusChip>
        </div>
        {liveError === undefined ? null : (
          <div className="notice notice-warning" role="alert">
            <strong>{liveError}</strong>
            <p>Use Refresh to reload the snapshot.</p>
          </div>
        )}
        <p className="ops-note">
          These are transport diagnostics. Runtime readiness and Run wait
          reasons are separate Server-owned facts. Refresh the snapshot if live
          updates are unavailable.
        </p>
      </div>
    </TechnicalDisclosure>
  );
}

/** What a session without the Operations capability sees under /operations. */
function OperationsCapabilityRequired() {
  return (
    <section
      className="ops-page ops-denied"
      aria-labelledby="operations-heading"
    >
      <div className="ops-denied-body">
        <span className="ops-denied-glyph">
          <StatusGlyph tone="blocked" size={22} />
        </span>
        <p className="ops-eyebrow">Operations capability required</p>
        <h1 id="operations-heading" className="ops-page-title">
          Operations
        </h1>
        <div className="notice notice-error" role="alert">
          This session is not authorized to observe or manage Operations
          resources.
        </div>
        <p className="ops-note">
          Your Git SSH key and theme stay available in{" "}
          <Link to={SETTINGS_PATH}>Settings</Link>.
        </p>
      </div>
    </section>
  );
}

export function OperationsLayoutRoute() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const location = useLocation();
  const { session } = useSession();
  const authorized =
    session?.principal.capabilities.includes("operations") === true;
  const normalizedPath = location.pathname.replace(/\/+$/, "");
  const personalSettings = !authorized && normalizedPath === SETTINGS_PATH;
  const independentRead =
    normalizedPath === "/operations/performance" ||
    normalizedPath === "/operations/allocations/completed";
  const currentSection = [...navigation]
    .reverse()
    .find(
      (item) =>
        normalizedPath === item.to || normalizedPath.startsWith(item.to + "/"),
    );
  useDocumentTitle(
    personalSettings
      ? "Settings"
      : currentSection === undefined || currentSection.to === "/operations"
        ? "Operations"
        : `${currentSection.label} · Operations`,
  );
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
    ...snapshotQuery(api),
    enabled: authorized && !independentRead,
  });
  // Bumped to recreate the live subscription, otherwise keyed by generation.
  const [liveEpoch, setLiveEpoch] = useState(0);
  const refresh = () => {
    // The event manager stops without a resync request after the Server
    // refuses the live session (close 1008). Refresh then subscribes again
    // from the new snapshot.
    const recreate = connection === "error";
    void query.refetch().then((result) => {
      if (recreate && result.isSuccess) setLiveEpoch((epoch) => epoch + 1);
    });
    for (const queryKey of [
      queryKeys.operations.runtimeAgentPrincipals.all,
      queryKeys.operations.runtimeConfigs.all,
      queryKeys.operations.runtimeLabels.all,
      queryKeys.operations.runtimeCredentials.all,
      queryKeys.configurations.all,
      queryKeys.credentials.all,
    ])
      void queryClient.invalidateQueries({ queryKey });
  };

  // Personal settings (Git SSH key, theme) are the one page without the
  // capability; it is reached from the account menu and draws its own page.
  if (personalSettings) return <Outlet />;
  if (!authorized) return <OperationsCapabilityRequired />;

  const liveKey = `${query.data?.cursor.generation ?? "none"}:${liveEpoch}`;
  return (
    <section className="ops-page" aria-labelledby="operations-heading">
      <header className="ops-page-head">
        <div className="ops-page-heading">
          <h1 id="operations-heading" className="ops-page-title">
            Operations
          </h1>
          <p className="ops-page-lede">
            Monitor Runtime Agents, inspect allocations and manage execution
            settings.
          </p>
        </div>
        {independentRead ? null : (
          <div className="ops-page-actions">
            <RefreshButton isFetching={query.isFetching} onRefresh={refresh} />
          </div>
        )}
      </header>

      <div className="ops-page-tabs">
        <nav
          className="ops-tabs section-navigation"
          aria-label="Operations sections"
        >
          {navigation.map((item) => (
            <Fragment key={item.to}>
              {item.to === SETUP_GROUP_START ? (
                <span className="ops-tabs-group" aria-hidden="true">
                  Setup
                </span>
              ) : null}
              <NavLink
                to={item.to}
                end={"end" in item ? item.end : false}
                className={({ isActive }) => (isActive ? "active" : undefined)}
              >
                {item.label}
              </NavLink>
            </Fragment>
          ))}
        </nav>
        <MobileSectionPicker
          label="Operations section"
          value={currentSection?.to ?? "/operations"}
          options={navigation}
          state={location.state}
        />
      </div>

      <div className="ops-page-body">
        {independentRead ? (
          <Outlet />
        ) : (
          <QueryView
            query={query}
            loading={
              <p className="ops-loading" role="status">
                Loading Operations snapshot…
              </p>
            }
            onRetry={refresh}
          >
            {(snapshot) => (
              <>
                <OperationsLiveSubscription
                  key={liveKey}
                  snapshot={snapshot}
                  onConnection={recordConnection}
                  onError={recordLiveError}
                  onResync={recordResync}
                />
                <Outlet
                  context={
                    {
                      snapshot: snapshot,
                      refresh,
                      refreshing: query.isFetching,
                    } satisfies OperationsOutletContext
                  }
                />
                <SnapshotDiagnostics
                  snapshot={snapshot}
                  connection={connection}
                  resyncReason={resyncReason}
                  liveError={liveError}
                />
              </>
            )}
          </QueryView>
        )}
      </div>
    </section>
  );
}
