import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { runtimeSettingsFromEnvironment } from "./runtime-config.mjs";
import { createStaticServer } from "./static-server.mjs";

const SHUTDOWN_TIMEOUT_MILLISECONDS = 5000;

export async function start(environment = process.env) {
  const settings = runtimeSettingsFromEnvironment(environment);
  const defaultDist = fileURLToPath(new URL("../dist", import.meta.url));
  const server = await createStaticServer({
    distDir: resolve(settings.distDir ?? defaultDist),
    runtimeConfig: settings.runtimeConfig,
  });
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(settings.port, settings.host, () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });
  process.stdout.write(
    `contractor-ui ${settings.runtimeConfig.uiVersion} listening on ${settings.host}:${settings.port}\n`,
  );

  let shuttingDown = false;
  const shutdown = () => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    const timer = setTimeout(
      () => server.closeAllConnections(),
      SHUTDOWN_TIMEOUT_MILLISECONDS,
    );
    timer.unref();
    server.close(() => {
      clearTimeout(timer);
      process.exitCode = 0;
    });
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
  return server;
}

const entrypoint = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (import.meta.url === entrypoint) {
  start().catch((error) => {
    const message = error instanceof Error ? error.message : "unknown failure";
    process.stderr.write(
      `contractor-ui: startup failed: ${message.slice(0, 512)}\n`,
    );
    process.exitCode = 1;
  });
}
