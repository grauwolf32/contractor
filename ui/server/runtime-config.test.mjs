import assert from "node:assert/strict";
import test from "node:test";

import {
  RuntimeConfigError,
  UI_VERSION,
  runtimeSettingsFromEnvironment,
  validateAPIBaseURL,
} from "./runtime-config.mjs";

test("runtime settings expose only the closed non-secret browser config", () => {
  const settings = runtimeSettingsFromEnvironment({
    CONTRACTOR_UI_API_BASE_URL: "https://api.example.test:8443",
    CONTRACTOR_UI_HOST: "0.0.0.0",
    CONTRACTOR_UI_PORT: "8080",
  });
  assert.deepEqual(settings.runtimeConfig, {
    uiVersion: UI_VERSION,
    supportedApiVersions: ["contractor.public.v1"],
    apiBaseUrl: "https://api.example.test:8443",
  });
  assert.equal(settings.host, "0.0.0.0");
  assert.equal(settings.port, 8080);
  assert.deepEqual(Object.keys(settings.runtimeConfig).sort(), [
    "apiBaseUrl",
    "supportedApiVersions",
    "uiVersion",
  ]);
});

test("runtime settings require a safe API origin", () => {
  assert.throws(() => runtimeSettingsFromEnvironment({}), RuntimeConfigError);
  for (const candidate of [
    "http://localhost:8080",
    "http://10.0.0.2:8080",
    "http://127.1:8080",
    "http://2130706433:8080",
    "http://0177.0.0.1:8080",
    "https://user:password@example.test",
    "https://example.test/v1",
    "https://example.test?credential=no",
    "https:/example.test",
    "https:\\example.test",
    "https://example.test/%2e%2e/",
  ]) {
    assert.throws(() => validateAPIBaseURL(candidate), RuntimeConfigError);
  }
  assert.equal(
    validateAPIBaseURL("http://127.99.1.2:8080/"),
    "http://127.99.1.2:8080",
  );
  assert.equal(validateAPIBaseURL("http://[::1]:8080"), "http://[::1]:8080");
});
