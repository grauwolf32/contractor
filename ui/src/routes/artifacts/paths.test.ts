import { describe, expect, it } from "vitest";

import { artifactDetailPath } from "./paths";

describe("artifactDetailPath", () => {
  const ref = { namespace: "inputs", name: "spec v1", revision: "r/1" };

  it("builds detail paths for every scope", () => {
    expect(artifactDetailPath({ kind: "user" }, ref)).toBe(
      "/artifacts/inputs/spec%20v1?revision=r%2F1",
    );
    expect(artifactDetailPath({ kind: "project", id: "project_a" }, ref)).toBe(
      "/projects/project_a/artifacts/inputs/spec%20v1?revision=r%2F1",
    );
    expect(
      artifactDetailPath({ kind: "project", id: "project_a" }, ref, "/evals"),
    ).toBe("/evals/project_a/artifacts/inputs/spec%20v1?revision=r%2F1");
    expect(artifactDetailPath({ kind: "run", id: "run-1" }, ref)).toBe(
      "/runs/run-1/artifacts/inputs/spec%20v1?revision=r%2F1",
    );
  });

  it("links to the current binding without a revision", () => {
    expect(
      artifactDetailPath({ kind: "user" }, { namespace: "a", name: "b" }),
    ).toBe("/artifacts/a/b");
  });
});
