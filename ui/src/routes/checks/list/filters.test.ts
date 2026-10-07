import { describe, expect, it } from "vitest";

import type { AuditState } from "../../../api/audits";
import {
  CHECK_FILTERS,
  checksHref,
  inFilter,
  parseCheck,
  parseCheckFilter,
  parseProject,
} from "./filters";

const STATES: AuditState[] = [
  "draft",
  "active",
  "waiting_review",
  "paused",
  "finalizing",
  "cancelling",
  "completed",
  "cancelled",
  "failed",
  "deleting",
];

describe("Checks list filters", () => {
  it("names the filters with the check state vocabulary", () => {
    expect(CHECK_FILTERS.map((option) => option.label)).toEqual([
      "Running",
      "Waiting for you",
      "Drafts",
      "Finished",
      "Stopped / failed",
      "All",
    ]);
  });

  it("puts every state in exactly one filter besides All", () => {
    for (const state of STATES) {
      const holding = CHECK_FILTERS.filter(
        (option) => option.value !== "all" && inFilter(option.value, state),
      ).map((option) => option.value);
      expect(holding.length).toBeLessThanOrEqual(1);
      expect(inFilter("all", state)).toBe(true);
      if (state !== "deleting") expect(holding).toHaveLength(1);
    }
    expect(inFilter("running", "paused")).toBe(true);
    expect(inFilter("stopped", "failed")).toBe(true);
  });

  it("reads filter names and check states from the URL", () => {
    expect(parseCheckFilter(null)).toBe("all");
    expect(parseCheckFilter("drafts")).toBe("drafts");
    expect(parseCheckFilter("waiting_review")).toBe("waiting");
    expect(parseCheckFilter("active")).toBe("running");
    expect(parseCheckFilter("cancelled")).toBe("stopped");
    expect(parseCheckFilter("completed")).toBe("finished");
    expect(parseCheckFilter("nonsense")).toBe("all");
    expect(parseProject("project_1")).toBe("project_1");
    expect(parseProject("../x")).toBeUndefined();
    expect(parseCheck("audit_1")).toBe("audit_1");
    expect(parseCheck("")).toBeUndefined();
  });

  it("builds list URLs that keep the filters and the selection", () => {
    expect(
      checksHref({ filter: "all", projectId: undefined, checkId: undefined }),
    ).toBe("/checks");
    expect(
      checksHref({ filter: "waiting", projectId: "p 1", checkId: "audit_1" }),
    ).toBe("/checks?state=waiting&project=p+1&check=audit_1");
  });
});
