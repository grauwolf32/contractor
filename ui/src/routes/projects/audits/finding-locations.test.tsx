import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { components } from "../../../api/generated/public";
import {
  isFindingLocation,
  validFindingCoordinates,
} from "../../../api/finding-locations";
import { decodeBody, isCredentialHeader } from "../../issues/evidence";
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

const exchange: NonNullable<Proposal["http_exchange"]> = {
  request_id: 42,
  request_tag: "probe",
  attempts: [
    {
      method: "POST",
      url: "https://target.test/login",
      headers: [
        { name: "Accept", value: "application/json" },
        { name: "Authorization", value: "Bearer request-secret" },
        { name: "Cookie", value: "sid=cookie-secret" },
        { name: "X-API-Key", value: "key-secret" },
        { name: "X-Auth-Token", value: "token-secret" },
      ],
      body_base64: btoa("user=alice"),
      status: 302,
      response_headers: [
        { name: "Set-Cookie", value: "sid=set-secret" },
        { name: "Location", value: "/home" },
      ],
    },
    {
      method: "GET",
      url: "https://target.test/home",
      headers: [],
      body_base64: btoa(String.fromCharCode(0, 1, 2, 255)),
      error: "transport_error",
    },
  ],
};

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
          http_exchange: exchange,
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
    expect(
      within(
        screen.getByRole("region", { name: "Finding locations" }),
      ).getAllByRole("listitem"),
    ).toHaveLength(4);
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
  it("masks credential-looking header values until each is shown", async () => {
    const { container } = render(
      <FindingLocations
        document={{ ...proposal, http_exchange: exchange }}
        exchangeOpen
      />,
    );
    expect(
      screen.getByText("Captured HTTP evidence · request 42"),
    ).toBeVisible();
    expect(screen.getByText("Attempt 1 of 2")).toBeVisible();
    expect(screen.getByText("POST https://target.test/login")).toBeVisible();
    expect(screen.getByText("HTTP 302")).toBeVisible();
    expect(screen.getByText("Transport error")).toBeVisible();
    expect(screen.getByText("application/json")).toBeVisible();
    expect(screen.getByText("/home")).toBeVisible();
    for (const secret of [
      "Bearer request-secret",
      "sid=cookie-secret",
      "key-secret",
      "token-secret",
      "sid=set-secret",
    ])
      expect(container).not.toHaveTextContent(secret);
    expect(screen.getAllByText("Hidden")).toHaveLength(5);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show Cookie value" }));
    expect(screen.getByText("sid=cookie-secret")).toBeVisible();
    expect(container).not.toHaveTextContent("Bearer request-secret");
    await user.click(screen.getByRole("button", { name: "Hide Cookie value" }));
    expect(container).not.toHaveTextContent("sid=cookie-secret");
    expect(
      screen.getByRole("button", { name: "Show Set-Cookie value" }),
    ).toBeVisible();
    // A readable body as text; bytes that are not text stay base64.
    expect(screen.getByText("user=alice")).toBeVisible();
    expect(
      screen.getByText(btoa(String.fromCharCode(0, 1, 2, 255))),
    ).toBeVisible();
  });
  it("names credential headers and decodes bodies without guessing", () => {
    for (const name of [
      "Authorization",
      "proxy-authorization",
      "Cookie",
      "Set-Cookie",
      "X-API-Key",
      "X-CSRF-Token",
      "x-refresh-token",
      "X-Client-Secret",
    ])
      expect(isCredentialHeader(name)).toBe(true);
    for (const name of ["Accept", "Content-Type", "Location", "User-Agent"])
      expect(isCredentialHeader(name)).toBe(false);
    expect(decodeBody("")).toEqual({ kind: "empty" });
    expect(decodeBody(btoa("a=1\n"))).toEqual({
      kind: "text",
      text: "a=1\n",
      bytes: 4,
    });
    expect(decodeBody(btoa(String.fromCharCode(0, 65)))).toEqual({
      kind: "binary",
      bytes: 2,
    });
    expect(decodeBody("not base64!")).toMatchObject({ kind: "binary" });
  });
});
