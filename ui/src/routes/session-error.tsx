import "./login.css";
import { useIsFetching, useQueryClient } from "@tanstack/react-query";

import { APICompatibilityError } from "../api/error";
import { queryKeys } from "../api/query-keys";
import { Button, StatusGlyph } from "../ui";
import { LoginBackdrop } from "./login-backdrop";
import { SignInBrand } from "./login-brand";

export function SessionConnectionError({ error }: { error: Error }) {
  const queryClient = useQueryClient();
  const refreshing =
    useIsFetching({ queryKey: queryKeys.session, exact: true }) > 0;
  const incompatible = error instanceof APICompatibilityError;

  return (
    <main className="ops-signin">
      <LoginBackdrop />
      <section
        className="ops-signin-card"
        aria-labelledby="session-error-heading"
      >
        <SignInBrand />
        <div className="ops-state-body">
          <p className="ops-state-eyebrow">
            <StatusGlyph tone="blocked" size={13} />
            Connection error
          </p>
          <h1 id="session-error-heading">
            {incompatible
              ? "Contractor Server is not compatible"
              : "Server unavailable"}
          </h1>
        </div>
        <p className="ops-state-message" role="alert">
          {error.message}
        </p>
        <div className="ops-state-actions">
          <Button
            variant="primary"
            pending={refreshing}
            pendingLabel="Loading…"
            onClick={() =>
              void queryClient.refetchQueries({
                queryKey: queryKeys.session,
                exact: true,
              })
            }
          >
            Try again
          </Button>
        </div>
      </section>
    </main>
  );
}
