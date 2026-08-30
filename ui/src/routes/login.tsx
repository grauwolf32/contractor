import { type FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router";

import { useSession } from "../auth/session";

function safeDestination(state: unknown): string {
  if (
    typeof state === "object" &&
    state !== null &&
    "from" in state &&
    typeof state.from === "string" &&
    state.from.startsWith("/") &&
    !state.from.startsWith("//")
  ) {
    return state.from;
  }
  return "/workflows";
}

export function LoginRoute() {
  const {
    session,
    error: sessionError,
    login,
    isLoggingIn,
    isLoading,
  } = useSession();
  const location = useLocation();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  if (session !== null && session !== undefined) {
    return <Navigate to={safeDestination(location.state)} replace />;
  }
  if (sessionError !== null) {
    return (
      <main className="centered-state">
        <p className="eyebrow">Connection error</p>
        <h1>Contractor Server is not compatible or unavailable</h1>
        <p role="alert">{sessionError.message}</p>
      </main>
    );
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    try {
      await login({ username, password });
      navigate(safeDestination(location.state), { replace: true });
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Authentication failed",
      );
    } finally {
      setPassword("");
    }
  }

  return (
    <main className="login-page">
      <section className="login-card" aria-labelledby="login-title">
        <div className="brand-mark" aria-hidden="true">
          C
        </div>
        <p className="eyebrow">Single-VM workflow runtime</p>
        <h1 id="login-title">Open the control workspace</h1>
        <p className="lede">
          Sign in with the local owner account configured on Contractor Server.
        </p>
        <form onSubmit={(event) => void onSubmit(event)}>
          <label>
            Username
            <input
              name="username"
              autoComplete="username"
              pattern="[A-Za-z0-9][A-Za-z0-9_.-]*"
              minLength={1}
              maxLength={64}
              required
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            Password
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              minLength={12}
              maxLength={1024}
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {error === null ? null : (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <button type="submit" disabled={isLoggingIn || isLoading}>
            {isLoggingIn ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}
