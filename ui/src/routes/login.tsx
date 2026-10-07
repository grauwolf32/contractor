import "./login.css";
import { useDocumentTitle } from "../app/document-title";
import { type FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router";

import { UI_VERSION } from "../build";
import { useSession } from "../auth/session";
import { StatusGlyph } from "../ui";
import { LoginBackdrop } from "./login-backdrop";
import { SignInBrand } from "./login-brand";
import { SessionConnectionError } from "./session-error";

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
  return "/";
}

export function LoginRoute() {
  useDocumentTitle("Sign in");
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
    return <SessionConnectionError error={sessionError} />;
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
    <main className="ops-signin">
      <LoginBackdrop />
      <section className="ops-signin-card" aria-label="Sign in">
        <SignInBrand />
        <h1>Sign in</h1>
        <form
          className="ops-signin-form"
          onSubmit={(event) => void onSubmit(event)}
        >
          <label>
            Username
            <input
              name="username"
              autoComplete="username"
              pattern={"[A-Za-z0-9][A-Za-z0-9_.\\-]*"}
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
              maxLength={1024}
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {error === null ? null : (
            <p className="ops-signin-error" role="alert">
              <StatusGlyph tone="blocked" size={15} />
              <span>{error}</span>
            </p>
          )}
          <button
            className="ui-btn"
            data-variant="primary"
            type="submit"
            disabled={isLoggingIn || isLoading}
          >
            {isLoggingIn ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <footer className="ops-signin-footer">
          <span className="ops-signin-version">UI {UI_VERSION}</span>
        </footer>
      </section>
    </main>
  );
}
