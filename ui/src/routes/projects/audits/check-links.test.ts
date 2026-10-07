import { describe, expect, it } from "vitest";

import {
  checkLinks,
  hashedItem,
  itemHash,
  parseSection,
  pinsItemList,
} from "./check-links";

const ROOT = "/projects/project_1/audits/audit_1";

describe("check links", () => {
  it("parses sections and item hashes", () => {
    expect(parseSection("coverage")).toBe("coverage");
    expect(parseSection("checks")).toBe("overview");
    expect(parseSection(undefined)).toBe("overview");
    expect(itemHash("item:1")).toBe("#check-item%3A1");
    expect(hashedItem("#check-item%3A1")).toBe("item:1");
    expect(hashedItem("#check-")).toBeUndefined();
    expect(hashedItem("#technical-details")).toBeUndefined();
    expect(hashedItem("#check-%E0%A4%A")).toBeUndefined();
    expect(pinsItemList("overview")).toBe(true);
    expect(pinsItemList("findings")).toBe(false);
  });

  it("carries list filters everywhere and the revision pin only along the list", () => {
    const current = new URLSearchParams(
      "result=issues&q=orders&auditRevision=4&state=pending&cursor=c1",
    );
    const onList = checkLinks("project_1", "audit_1", current, "coverage");
    expect(onList.item("item_1")).toEqual({
      pathname: `${ROOT}/coverage`,
      search: "?result=issues&q=orders&auditRevision=4",
      hash: "#check-item_1",
    });
    expect(onList.overview).toEqual({
      pathname: ROOT,
      search: "?result=issues&q=orders&auditRevision=4",
    });
    expect(onList.section("findings")).toEqual({
      pathname: `${ROOT}/findings`,
      search: "?result=issues&q=orders",
    });
    expect(onList.group("all")).toEqual({
      pathname: `${ROOT}/coverage`,
      search: "?q=orders",
    });
    expect(onList.group("uncertain", 7)).toEqual({
      pathname: `${ROOT}/coverage`,
      search: "?result=uncertain&q=orders&auditRevision=7",
    });
    expect(
      onList.deep("reviews", { state: "pending", auditRevision: "4" }),
    ).toEqual({
      pathname: `${ROOT}/reviews`,
      search: "?result=issues&q=orders&state=pending&auditRevision=4",
    });
    // On a queue section the revision pins the queue, not the list.
    const onQueue = checkLinks("project_1", "audit_1", current, "reviews");
    expect(onQueue.list).toEqual({
      pathname: `${ROOT}/coverage`,
      search: "?result=issues&q=orders",
    });
    expect(
      checkLinks("project_1", "audit_1", new URLSearchParams(), "overview")
        .list,
    ).toEqual({ pathname: `${ROOT}/coverage`, search: "" });
  });
});
