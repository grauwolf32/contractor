import cases from "../../../api/testdata/model-profile-id-cases.json";
import { describe, expect, it } from "vitest";
import { schemaViolation } from "./contract";

describe("shared model profile ID grammar", () => {
  it.each(cases.valid)("accepts %j", (id) => {
    expect(schemaViolation("ModelProfileId", id)).toBeUndefined();
    expect(
      schemaViolation("ModelSelectionValue", {
        defaultModel: id,
        roles: { summarizer: id },
      }),
    ).toBeUndefined();
  });
  it.each(cases.invalid)("rejects %j", (id) => {
    expect(schemaViolation("ModelProfileId", id)).toBeDefined();
    expect(
      schemaViolation("ModelSelectionValue", {
        defaultModel: "fixture",
        roles: { worker: id },
      }),
    ).toBeDefined();
  });
});
