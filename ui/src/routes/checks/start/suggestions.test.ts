import { describe, expect, it } from "vitest";

import {
  FORMAT_MATCH_REASON,
  SUGGESTION_RULES,
  suggestCheckTypes,
} from "./suggestions";

const everything = () => true;
const only =
  (...names: string[]) =>
  (name: string) =>
    names.includes(name);

describe("check type suggestions", () => {
  it("suggests nothing for a blank objective", () => {
    expect(suggestCheckTypes("   ", everything)).toEqual({});
  });

  it("maps words in the objective to a ready check type", () => {
    const result = suggestCheckTypes(
      "Check the shop and mechanic APIs for authorization flaws",
      everything,
    );
    expect(result.suggestion?.checkType).toBe("openapi-operation-trace");
    expect(result.suggestion?.ruleId).toBe("api-access");
    // "flaws" also matches the broad rule, which becomes the alternative.
    expect(result.alternative?.checkType).toBe("owasp-top10-2025-source-risk");
  });

  it("never suggests a check type the project's materials cannot start", () => {
    const result = suggestCheckTypes(
      "Find IDOR in the API",
      only("owasp-top10-2025-source-risk"),
    );
    expect(result.suggestion).toBeUndefined();
    expect(
      suggestCheckTypes(
        "Find IDOR in the API",
        only("openapi-operation-observe"),
      ).suggestion?.checkType,
    ).toBe("openapi-operation-observe");
  });

  it("matches whole words only, in any case", () => {
    expect(
      suggestCheckTypes("Rapid review of capital letters", everything)
        .suggestion,
    ).toBeUndefined();
    expect(
      suggestCheckTypes("verify ASVS LEVEL 1", everything).suggestion
        ?.checkType,
    ).toBe("owasp-asvs-5-0-l1-source-review");
  });

  it("orders specific rules before broad ones", () => {
    expect(
      suggestCheckTypes("SQL injection on the staging website", everything)
        .suggestion?.checkType,
    ).toBe("openapi-sqlmap-scan");
    expect(
      suggestCheckTypes(
        "SQL injection on the staging website",
        (name) => name !== "openapi-sqlmap-scan",
      ).suggestion?.checkType,
    ).toBe("owasp-wstg-4-2-active-http");
  });

  it("gives the rule's fixed reason and says only that formats match", () => {
    const result = suggestCheckTypes("security risks", everything);
    const rule = SUGGESTION_RULES.find((item) => item.id === "security-risks");
    expect(result.suggestion?.reason).toBe(
      `${rule?.reason} ${FORMAT_MATCH_REASON}`,
    );
    expect(FORMAT_MATCH_REASON).toMatch(/format matches/);
    for (const item of SUGGESTION_RULES) {
      expect(item.reason).toMatch(/^Your objective mentions /);
      expect(item.reason).not.toMatch(/\b(?:proven|certif|guarantee)/i);
    }
  });

  it("offers an alternative from a different check type only", () => {
    const result = suggestCheckTypes(
      "API endpoints",
      only("openapi-operation-trace"),
    );
    expect(result.suggestion?.checkType).toBe("openapi-operation-trace");
    expect(result.alternative).toBeUndefined();
  });
});
