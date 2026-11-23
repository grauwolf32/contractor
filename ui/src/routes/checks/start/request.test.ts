import { describe, expect, it } from "vitest";

import { createRequest, parseRuntimeLabels } from "./request";

describe("create request", () => {
  it("parses runtime labels separated by commas or spaces, once each", () => {
    expect(parseRuntimeLabels(" debug, caido  debug\ncaido-2 ")).toEqual({
      labels: ["debug", "caido", "caido-2"],
      invalid: [],
    });
    expect(parseRuntimeLabels("Debug, ok, 9lives")).toEqual({
      labels: ["ok"],
      invalid: ["Debug", "9lives"],
    });
  });

  it("sends trimmed scope fields that are not empty and no empty lists", () => {
    expect(
      createRequest({
        profile: { name: "source-checklist", version: "1" },
        inputs: {
          source: { namespace: "sources", name: "app", revision: "r1" },
        },
        objective: "  Map attack surface ",
        target: "",
        authorizationScope: "   ",
        runtimeLabels: [],
      }),
    ).toEqual({
      profile: { name: "source-checklist", version: "1" },
      inputs: { source: { namespace: "sources", name: "app", revision: "r1" } },
      scope: { objective: "Map attack surface" },
    });
  });

  it("includes every scope field and runtime label it has", () => {
    expect(
      createRequest({
        profile: { name: "wstg", version: "2" },
        inputs: {},
        objective: "",
        target: "https://staging.example.com",
        authorizationScope: "Staging only",
        runtimeLabels: ["caido"],
      }),
    ).toEqual({
      profile: { name: "wstg", version: "2" },
      inputs: {},
      runtimeLabels: ["caido"],
      scope: {
        target: "https://staging.example.com",
        authorizationScope: "Staging only",
      },
    });
  });
});
