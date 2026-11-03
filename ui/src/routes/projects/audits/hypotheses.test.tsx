import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { fakeServer } from "../../../test/fake-server";
import { json, failure } from "../../../test/responses";
import { renderWithAPI } from "../../../test/render";
import { makeAudit } from "./check-test-support";
import { AuditHypotheses } from "./hypotheses";

function fixture(rejectFirst = 0) {
  const audit = makeAudit("audit-one", "project-one", "paused");
  const writes: {
    body: unknown;
    key: string | null;
    revision: string | null;
    path: string;
  }[] = [];
  const server = fakeServer({
    "GET /v1/audits/{auditId}/hypotheses": () =>
      json({
        checking_workflows: ["inspect"],
        fixed_denominator: 2,
        pending_fixed_count: 0,
        items: [],
        page: { hasMore: false },
      }),
    "POST /v1/audits/{auditId}/hypotheses": async ({ request, url }) => {
      writes.push({
        body: await request.json(),
        key: request.headers.get("Idempotency-Key"),
        revision: request.headers.get("If-Match"),
        path: url.pathname,
      });
      if (rejectFirst && writes.length === 1)
        return failure(
          rejectFirst,
          rejectFirst === 412
            ? "precondition_failed"
            : "storage_transaction_conflict",
        );
      return json({
        hypotheses: [{ id: "hypothesis-one", revision: 1 }],
        audit_revision: 4,
      });
    },
    "POST /v1/audits/{auditId}/tasks": async ({ request, url }) => {
      writes.push({
        body: await request.json(),
        key: request.headers.get("Idempotency-Key"),
        revision: request.headers.get("If-Match"),
        path: url.pathname,
      });
      return json({
        task_ids: ["task-one"],
        created_count: 1,
        audit_revision: 4,
      });
    },
  });
  renderWithAPI(<AuditHypotheses audit={audit} />, { server });
  return writes;
}
async function fill() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Show assertions" }));
  await screen.findByText(/2 requirements in the original checklist/);
  await user.type(
    screen.getByRole("textbox", { name: /^Assertion/ }),
    "The selected actor can read another record",
  );
  await user.type(
    screen.getByRole("textbox", { name: /^Conditions/ }),
    "Selected fixture only",
  );
  await user.type(
    screen.getByRole("textbox", { name: /^Why check this/ }),
    "Analyst observation",
  );
  return user;
}
describe("assertion commands", () => {
  it("retries an ambiguous save with the identical key, content and revision", async () => {
    const writes = fixture(503);
    const user = await fill();
    await user.click(screen.getByRole("button", { name: "Save assertion" }));
    await screen.findByRole("button", { name: "Retry save" });
    expect(screen.getByRole("textbox", { name: /^Assertion/ })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Retry save" }));
    await screen.findByText(/Assertion saved\./);
    expect(writes).toHaveLength(2);
    expect(writes[1]).toEqual(writes[0]);
    expect(writes[0]!.revision).toBe('"3"');
  });
  it("saves the assertion and accepts its check in one Task command", async () => {
    const writes = fixture();
    const user = await fill();
    await user.click(
      await screen.findByRole("button", { name: "Save and check" }),
    );
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0]!.path).toBe("/v1/audits/audit-one/tasks");
    const body = writes[0]!.body as {
      work: { key: string }[];
      assertions: Record<string, { statement: string; scope: unknown }>;
      attention: string[];
    };
    const key = body.work[0]!.key;
    expect(body.attention).toEqual([key]);
    expect(body.assertions[key]).toMatchObject({
      statement: "The selected actor can read another record",
      scope: { conditions: "Selected fixture only" },
    });
  });
  it("keeps a new assertion editable after a definite stale-revision refusal", async () => {
    const writes = fixture(412);
    const user = await fill();
    await user.click(screen.getByRole("button", { name: "Save assertion" }));
    await screen.findByText(/This check changed/);
    expect(
      screen.getByRole("button", { name: "Save assertion" }),
    ).toBeDisabled();
    expect(screen.getByRole("textbox", { name: /^Assertion/ })).toHaveValue(
      "The selected actor can read another record",
    );
    expect(writes).toHaveLength(1);
    await user.click(
      screen.getByRole("button", { name: "Use refreshed revision" }),
    );
    expect(
      screen.getByRole("button", { name: "Save assertion" }),
    ).toBeEnabled();
    expect(writes).toHaveLength(1);
  });
});
