import { screen, within } from "@testing-library/react";
import { expect, it } from "vitest";

import memberPage from "../../../../api/testdata/evals/valid/member-page.json";
import type { EvalMember } from "../../api/evals";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { json } from "../../test/responses";
import { MemberExecutions } from "./member";

it("shows a common iteration but omits it for a mixed Task batch", async () => {
  const member = memberPage.items[0] as EvalMember;
  renderWithAPI(<MemberExecutions id="experiment-1" member={member} />, {
    server: fakeServer({
      "GET /v1/eval-experiments/{id}/members/{memberId}/executions": () =>
        json({
          inventoryRevision: 1,
          inventoryComplete: true,
          items: [1, null].map((iteration, index) => ({
            execution: { kind: "run", id: `run-${index}` },
            parent: { kind: "audit", id: "audit-1" },
            workflow: "observe",
            iteration,
            state: "succeeded",
            available: true,
          })),
          gaps: [],
          page: { hasMore: false, nextCursor: null },
        }),
    }),
  });
  const uniform = (
    await screen.findByRole("link", { name: "Run run-0" })
  ).closest("li")!;
  const mixed = screen.getByRole("link", { name: "Run run-1" }).closest("li")!;
  expect(
    within(uniform).getByText("observe · iteration 1 · succeeded"),
  ).toBeVisible();
  expect(within(mixed).getByText("observe · succeeded")).toBeVisible();
  expect(mixed).not.toHaveTextContent(/iteration/);
});
