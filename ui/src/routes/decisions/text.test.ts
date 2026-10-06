import { describe, expect, it } from "vitest";

import { makeFinding } from "./test-support";
import {
  aiSummary,
  firstParagraph,
  httpOperation,
  plainText,
  splitImpact,
  standardReferences,
  weaknessReferences,
} from "./text";

describe("splitImpact", () => {
  it("moves an Impact heading and its subsections into their own part", () => {
    expect(
      splitImpact(
        [
          "Found it.",
          "",
          "## Impact",
          "",
          "Anyone can read orders.",
          "",
          "### Data",
          "",
          "Email and phone.",
          "",
          "## Fix",
          "",
          "Scope the query.",
        ].join("\n"),
      ),
    ).toEqual({
      found: "Found it.\n\n## Fix\n\nScope the query.",
      impact: "Anyone can read orders.\n\n### Data\n\nEmail and phone.",
    });
  });

  it("reads a bold Impact label at the start of a paragraph", () => {
    expect(
      splitImpact(
        "Found it.\n\n**Impact:** Anyone can\nread orders.\n\nMore detail.",
      ),
    ).toEqual({
      found: "Found it.\n\nMore detail.",
      impact: "Anyone can\nread orders.",
    });
    expect(splitImpact("**Potential impact**: data loss.").impact).toBe(
      "data loss.",
    );
  });

  it("leaves code blocks and empty Impact sections alone", () => {
    const fenced = "Found it.\n\n```md\n## Impact\n\nnot a heading\n```";
    expect(splitImpact(fenced)).toEqual({ found: fenced });
    expect(splitImpact("Found it.\n\n## Impact\n\n## Fix\n\nDo X.")).toEqual({
      found: "Found it.\n\n## Impact\n\n## Fix\n\nDo X.",
    });
  });
});

describe("firstParagraph", () => {
  it("skips headings and keeps fenced code whole", () => {
    expect(firstParagraph("# Title\n\nFirst line\nsecond line\n\nNext.")).toBe(
      "First line\nsecond line",
    );
    expect(
      firstParagraph("```js\nconst a = 1;\n\nconst b = 2;\n```\n\nText"),
    ).toBe("```js\nconst a = 1;\n\nconst b = 2;\n```");
    expect(firstParagraph("## Only a heading")).toBe("");
  });
});

describe("plainText", () => {
  it("keeps the words of Markdown", () => {
    expect(
      plainText(
        "## Head\n\n- The **bold** and _soft_ [link](https://x.test) to `code`\n> quoted <b>tag</b>",
      ),
    ).toBe("Head The bold and soft link to code quoted tag");
    expect(plainText("snake_case_name stays")).toBe("snake_case_name stays");
  });
});

describe("aiSummary", () => {
  it("joins the title and the first sentence of what the AI found", () => {
    expect(
      aiSummary(
        makeFinding(
          {},
          {
            title: "IDOR on reports",
            description:
              "`GetReportView.get` in views.py is protected only by a sign-in check. The query is not scoped.\n\n## Impact\n\nAll reports leak.",
          },
        ),
      ),
    ).toBe(
      "IDOR on reports. GetReportView.get in views.py is protected only by a sign-in check.",
    );
  });

  it("falls back to the title and stays short", () => {
    expect(
      aiSummary(makeFinding({}, { title: "Weak hash!", description: "" })),
    ).toBe("Weak hash!");
    const long = aiSummary(
      makeFinding({}, { description: `${"word ".repeat(120)}end.` }),
    );
    expect(long.length).toBeLessThanOrEqual(300);
    expect(long.endsWith("…")).toBe(true);
  });
});

describe("references", () => {
  it("reads HTTP operations from subject keys", () => {
    expect(httpOperation("get /orders/{id}")).toEqual({
      method: "GET",
      path: "/orders/{id}",
    });
    expect(httpOperation("POST https://api.test/v1/x")).toEqual({
      method: "POST",
      path: "https://api.test/v1/x",
    });
    expect(httpOperation("get-widget")).toBeUndefined();
    expect(httpOperation("GET orders")).toBeUndefined();
  });

  it("separates weaknesses from other standards", () => {
    const document = {
      standard_refs: [
        { scheme: "CWE", version: "4.20", requirement_id: "CWE-639" },
        { scheme: "asvs", version: "4.0.3", requirement_id: "V4.1.1" },
      ],
    };
    expect(
      weaknessReferences(document).map((ref) => ref.requirement_id),
    ).toEqual(["CWE-639"]);
    expect(
      standardReferences(document).map((ref) => ref.requirement_id),
    ).toEqual(["V4.1.1"]);
  });
});
