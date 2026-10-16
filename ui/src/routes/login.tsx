import "./login.css";
import { useDocumentTitle } from "../app/document-title";
import { type FormEvent, useState } from "react";
import { Navigate, useLocation } from "react-router";

import { htmlPattern, stringRule } from "../api/contract";
import { useGuardedNavigate } from "../app/page-location";
import { UI_VERSION } from "../build";
import { useSession } from "../auth/session";
import { Button, StatusGlyph } from "../ui";
import { LoginBackdrop } from "./login-backdrop";
import { SignInBrand } from "./login-brand";
import { SessionConnectionError } from "./session-error";

// The sign-in form's limits are the LoginRequest schema's.
const USERNAME = stringRule("LoginRequest", "properties", "username");
const USERNAME_PATTERN = htmlPattern(USERNAME);
const PASSWORD = stringRule("LoginRequest", "properties", "password");

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
  const navigate = useGuardedNavigate();
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
      void navigate(safeDestination(location.state), { replace: true });
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
              pattern={USERNAME_PATTERN}
              minLength={USERNAME.minLength}
              maxLength={USERNAME.maxLength}
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
              maxLength={PASSWORD.maxBytes}
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
          <Button
            variant="primary"
            type="submit"
            disabled={isLoading}
            pending={isLoggingIn}
            pendingLabel="Signing in…"
          >
            Sign in
          </Button>
        </form>
        <footer className="ops-signin-footer">
          <span className="ops-signin-version">UI {UI_VERSION}</span>
        </footer>
      </section>
    </main>
  );
}
