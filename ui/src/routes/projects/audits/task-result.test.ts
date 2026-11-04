import { describe, expect, it } from "vitest";
import { taskResultPreview } from "./task-result";

describe("saved batch result", () => {
  it("shows only the trusted Task member, preserving independent verdicts", () => {
    const text = JSON.stringify({
      schema: "contractor.audit.result.v1",
      items: [
        {
          verdict: "supported",
          origin: { task_id: "a", hypothesis: { id: "first", revision: 1 } },
        },
        {
          verdict: "refuted",
          origin: { task_id: "b", hypothesis: { id: "second", revision: 2 } },
        },
      ],
    });
    expect(JSON.parse(taskResultPreview(text, "a"))).toEqual({
      verdict: "supported",
      origin: { task_id: "a", hypothesis: { id: "first", revision: 1 } },
    });
    expect(taskResultPreview(text, "a")).not.toContain("refuted");
    expect(() => taskResultPreview(text, "absent")).toThrow("unique member");
  });
});
