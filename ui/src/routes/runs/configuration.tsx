import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { Navigate, Outlet, useLocation } from "react-router";

import { usePublicAPI } from "../../api/context";
import { getOperationsSnapshot } from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { useSession } from "../../auth/session";
import { useRunEvents } from "../../events/context";
import { ErrorNotice } from "../artifacts/common";

export function RunConfigurationLayout() {
  const api = usePublicAPI();
  const events = useRunEvents();
  const queryClient = useQueryClient();
  const [liveError, setLiveError] = useState<string>();
  const { session } = useSession();
  const authorized =
    session?.principal.capabilities.includes("operations") === true;
  const snapshot = useQuery({
    queryKey: queryKeys.operations.snapshot,
    queryFn: () => getOperationsSnapshot(api),
    enabled: authorized,
  });

  const refresh = useCallback(() => {
    for (const queryKey of [
      queryKeys.operations.runtimeConfigs.all,
      queryKeys.operations.runtimeLabels.all,
      queryKeys.operations.runtimeCredentials.all,
      queryKeys.configurations.all,
    ])
      void queryClient.invalidateQueries({ queryKey });
  }, [queryClient]);
  useEffect(() => {
    if (!authorized || snapshot.data === undefined) return;
    const subscription = events.subscribeOperations(snapshot.data.cursor, {
      onOperationsEvent: refresh,
      onResync: () => {
        refresh();
        void queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        });
      },
      onStateChange: (state) => {
        if (state === "live") {
          setLiveError(undefined);
          refresh();
        }
      },
      onError: (message) => setLiveError(message),
    });
    return () => subscription.unsubscribe();
  }, [authorized, events, queryClient, refresh, snapshot.data]);

  if (!authorized) {
    return (
      <div className="notice notice-error" role="alert">
        Operations capability is required to manage Runtime configurations.
      </div>
    );
  }
  return (
    <>
      <div className="section-heading">
        <h3>Runtime configuration</h3>
        <button
          className="secondary-button"
          type="button"
          onClick={() => {
            refresh();
            void snapshot.refetch();
          }}
        >
          Refresh configuration
        </button>
      </div>
      {snapshot.error === null ? null : <ErrorNotice error={snapshot.error} />}
      {liveError === undefined ? null : (
        <div className="notice notice-warning" role="status">
          {liveError} Manual configuration refresh remains available.
        </div>
      )}
      <Outlet />
    </>
  );
}

export function LegacyRuntimeConfigurationRedirect() {
  const location = useLocation();
  return (
    <Navigate
      replace
      to={{
        pathname: location.pathname.replace(
          /^\/operations\/runtime-configs/,
          "/runs/configuration",
        ),
        search: location.search,
        hash: location.hash,
      }}
    />
  );
}
