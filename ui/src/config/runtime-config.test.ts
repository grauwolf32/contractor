import { describe, expect, it, vi } from "vitest";

import { UI_VERSION } from "../build";
import {
  loadRuntimeConfig,
  parseRuntimeConfig,
  RuntimeConfigError,
  validateAPIBaseURL,
} from "./runtime-config";

const validConfig = {
  uiVersion: UI_VERSION,
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "https://api.contractor.example:8443",
};

describe("runtime configuration", () => {
  it("accepts only the exact compatible closed shape", () => {
    expect(parseRuntimeConfig(validConfig)).toEqual(validConfig);
    expect(() => parseRuntimeConfig({ ...validConfig, secret: "no" })).toThrow(
      RuntimeConfigError,
    );
    expect(() =>
      parseRuntimeConfig({ ...validConfig, uiVersion: "another-build" }),
    ).toThrow("UI build and runtime configuration differ");
    expect(() =>
      parseRuntimeConfig({ ...validConfig, supportedApiVersions: ["v2"] }),
    ).toThrow("incompatible APIs");
  });

  it.each([
    "http://localhost:8080",
    "http://192.0.2.20:8080",
    "http://127.1:8080",
    "http://2130706433:8080",
    "http://0177.0.0.1:8080",
    "https://user@example.test",
    "https://example.test/v1",
    "https://example.test?token=no",
    "https://example.test/#fragment",
    "https:/example.test",
    "https:\\example.test",
    "https://example.test/%2e%2e/",
    " https://example.test",
  ])("rejects unsafe API origin %s", (candidate) => {
    expect(() => validateAPIBaseURL(candidate)).toThrow(RuntimeConfigError);
  });

  it.each([
    ["https://api.example.test", "https://api.example.test"],
    ["http://127.0.0.1:8080", "http://127.0.0.1:8080"],
    ["http://127.12.34.56:8080/", "http://127.12.34.56:8080"],
    ["http://[::1]:8080", "http://[::1]:8080"],
    ["http://10.20.30.40:8080", "http://10.20.30.40:8080"],
    ["http://172.16.1.2:8080", "http://172.16.1.2:8080"],
    ["http://192.168.1.217:8080", "http://192.168.1.217:8080"],
  ])("normalizes safe API origin %s", (candidate, expected) => {
    expect(validateAPIBaseURL(candidate)).toBe(expected);
  });

  it("loads runtime config without credentials or caching", async () => {
    const fetchImplementation = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(validConfig), {
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(loadRuntimeConfig(fetchImplementation)).resolves.toEqual(
      validConfig,
    );
    expect(fetchImplementation).toHaveBeenCalledWith(
      "/runtime-config.json",
      expect.objectContaining({ credentials: "omit", cache: "no-store" }),
    );
  });

  it("rejects an oversized runtime config before parsing", async () => {
    const fetchImplementation = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        new Response("{}", { headers: { "content-length": "9000" } }),
      );
    await expect(loadRuntimeConfig(fetchImplementation)).rejects.toThrow(
      "too large",
    );
  });

  it("stops reading an oversized streamed runtime config", async () => {
    const fetchImplementation = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("x".repeat(9000)));
    await expect(loadRuntimeConfig(fetchImplementation)).rejects.toThrow(
      "too large",
    );
  });
});
