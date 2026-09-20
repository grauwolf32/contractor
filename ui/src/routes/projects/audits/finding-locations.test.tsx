import { render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { components } from "../../../api/generated/public";
import {
  isFindingLocation,
  validFindingCoordinates,
} from "../../../api/finding-locations";
import { FindingLocations } from "./finding-locations";

type Proposal = components["schemas"]["FindingProposalDocument"];
const proposal: Proposal = {
  schema: "contractor.audit.finding-proposal.v1",
  client_key: "call-1",
  title: "Observed issue",
  description: "A source observation",
  subject: null,
  preconditions: [],
  standard_refs: [],
  evidence_ids: [],
  proposed_checks: [],
  severity_suggestion: "",
  limitations: [],
};
const cases = JSON.parse(
  readFileSync(
    "../internal/auditdomain/testdata/finding-locations.json",
    "utf8",
  ),
) as { name: string; location: unknown; valid: boolean }[];

describe("finding locations", () => {
  it.each(cases)(
    "matches the Go/Python coordinate contract: $name",
    ({ location, valid }) => {
      expect(isFindingLocation(location)).toBe(valid);
    },
  );
  it("renders authored paths and URLs without creating links or inferring coordinates", () => {
    render(
      <FindingLocations
        document={{
          ...proposal,
          locations: [
            { file: "src/auth.py" },
            { file: "src/order.py", line: 42 },
            { file: "src/check.py", range: { start_line: 10, end_line: 12 } },
            { url: "https://target.test/a?x=%2f&x=2#section", method: "POST" },
          ],
        }}
      />,
    );
    expect(screen.getByText("src/auth.py")).toBeInTheDocument();
    expect(screen.getByText("src/order.py:42")).toBeInTheDocument();
    expect(screen.getByText("src/check.py:10–12")).toBeInTheDocument();
    expect(
      screen.getByText("https://target.test/a?x=%2f&x=2#section"),
    ).toBeInTheDocument();
    expect(screen.queryAllByRole("link")).toHaveLength(0);
  });
  it("permits absent coordinates and rejects malformed or unsupported-schema extensions", () => {
    expect(validFindingCoordinates(proposal)).toBe(true);
    expect(validFindingCoordinates({ ...proposal, locations: null })).toBe(
      false,
    );
    expect(
      validFindingCoordinates({
        ...proposal,
        schema: "contractor.audit.finding-proposal.unsupported",
        locations: [{ file: "a.py" }],
      }),
    ).toBe(false);
    const { container } = render(<FindingLocations document={proposal} />);
    expect(container).toBeEmptyDOMElement();
  });
});
