import { useIsFetching, useQueryClient } from "@tanstack/react-query";

import { APICompatibilityError } from "../api/error";
import { queryKeys } from "../api/query-keys";

export function SessionConnectionError({ error }: { error: Error }) {
  const queryClient = useQueryClient();
  const refreshing =
    useIsFetching({ queryKey: queryKeys.session, exact: true }) > 0;

  return (
    <main className="centered-state">
      <p className="eyebrow">Connection error</p>
      <h1>
        {error instanceof APICompatibilityError
          ? "Contractor Server is not compatible"
          : "Server unavailable"}
      </h1>
      <p role="alert">{error.message}</p>
      <button
        className="secondary-button"
        type="button"
        disabled={refreshing}
        onClick={() =>
          void queryClient.refetchQueries({
            queryKey: queryKeys.session,
            exact: true,
          })
        }
      >
        {refreshing ? "Loading…" : "Try again"}
      </button>
    </main>
  );
}
