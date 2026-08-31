import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { PublicAPI } from "./api/client";
import { Application } from "./app/application";
import { createApplicationRouter } from "./app/router";
import { loadRuntimeConfig } from "./config/runtime-config";
import "./styles.css";

function renderBootstrapFailure(root: HTMLElement, error: unknown): void {
  const message =
    error instanceof Error
      ? error.message
      : "Contractor UI could not read its runtime configuration";
  createRoot(root).render(
    <StrictMode>
      <main className="centered-state">
        <p className="eyebrow">Startup blocked</p>
        <h1>UI runtime configuration is invalid</h1>
        <p role="alert">{message}</p>
      </main>
    </StrictMode>,
  );
}

async function bootstrap(): Promise<void> {
  const root = document.getElementById("root");
  if (root === null) {
    throw new Error("UI root element is missing");
  }
  try {
    const runtimeConfig = await loadRuntimeConfig();
    const api = new PublicAPI(runtimeConfig);
    const router = createApplicationRouter();
    createRoot(root).render(
      <StrictMode>
        <Application api={api} publicAPI={api} router={router} />
      </StrictMode>,
    );
  } catch (error) {
    renderBootstrapFailure(root, error);
  }
}

void bootstrap();
