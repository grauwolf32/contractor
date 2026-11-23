import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import { renderApplication } from "../../../test/application";
import { audit, conforming } from "../../../test/builders";
import { fakeServer } from "../../../test/fake-server";
import { json, page } from "../../../test/responses";

it("reads an ordinary retained task result without a hypothesis or Finding", async () => {
  const check = audit("project_1", "audit_1", "completed", {
    taskCount: 1,
    reservedRunCount: 1,
    submittedRunCount: 1,
  });
  const task = conforming("AuditTask", {
    attention: false,
    document: {
      schema: "contractor.audit.task.v1",
      audit_id: "audit_1",
      task_id: "task_1",
      workflow: "inspect",
      objective: "Inspect selected material",
      inputs: {},
      parameters: {},
    },
    source: { kind: "manual", key: "manual-1" },
    ordinal: 1,
    iteration: 1,
    state: "settled",
    attempts: 1,
    last_execution_id: "execution_1",
    result: {
      scope_kind: "audit",
      scope_id: "audit_1",
      ref: { namespace: "saved", name: "result", revision: "result-r1" },
      digest: `sha256:${"a".repeat(64)}`,
      media_type: "application/json",
      size_bytes: 2,
    },
    created_at: check.createdAt,
    updated_at: check.updatedAt,
  });
  const payload =
    '{"rationale":"Selected bytes inspected","limitations":["No live traffic"]}';
  const server = fakeServer({
    "GET /v1/audits/{auditId}": () =>
      json(check, 200, { ETag: `"${check.revision}"` }),
    "GET /v1/audits/{auditId}/tasks": () => json(page([task])),
    "GET /v1/audits/{auditId}/artifacts": () =>
      new Response(payload, {
        headers: {
          "X-Contractor-API-Version": "contractor.public.v1",
          "Content-Type": "application/json",
          "Content-Length": String(new TextEncoder().encode(payload).length),
        },
      }),
  });
  renderApplication("/projects/project_1/audits/audit_1", { server });
  expect(
    await screen.findByRole("heading", { name: "Inspect selected material" }),
  ).toBeInTheDocument();
  await userEvent.click(
    screen.getByRole("button", { name: "Read saved result" }),
  );
  expect(
    await screen.findByText(/Selected bytes inspected/),
  ).toBeInTheDocument();
  expect(server.sent("GET", "/artifacts")[0]?.url).toContain(
    "key=task%2Ftask_1%2Fresult",
  );
  expect(server.sent("GET", "/coverage")).toHaveLength(0);
  expect(server.sent("GET", "/report")).toHaveLength(0);
});

it("pages accepted tasks instead of stopping after the first page", async () => {
  const check = audit("project_1", "audit_1", "paused", { taskCount: 2 });
  const server = fakeServer({
    "GET /v1/audits/{auditId}": () =>
      json(check, 200, { ETag: `"${check.revision}"` }),
    "GET /v1/audits/{auditId}/tasks": ({ url }) =>
      json(
        page(
          [
            conforming("AuditTask", {
              attention: false,
              document: {
                schema: "contractor.audit.task.v1",
                audit_id: "audit_1",
                task_id: url.searchParams.has("cursor") ? "task_2" : "task_1",
                workflow: "inspect",
                objective: url.searchParams.has("cursor")
                  ? "Second task"
                  : "First task",
                inputs: {},
                parameters: {},
              },
              source: {
                kind: "manual",
                key: url.searchParams.has("cursor") ? "second" : "first",
              },
              ordinal: url.searchParams.has("cursor") ? 2 : 1,
              iteration: 1,
              state: "ready",
              attempts: 0,
              created_at: check.createdAt,
              updated_at: check.updatedAt,
            }),
          ],
          url.searchParams.has("cursor") ? undefined : "page-2",
        ),
      ),
  });
  renderApplication("/projects/project_1/audits/audit_1", { server });
  await userEvent.click(await screen.findByRole("button", { name: "Next" }));
  expect(
    await screen.findByRole("heading", { name: "Second task" }),
  ).toBeInTheDocument();
  expect(
    server
      .sent("GET", "/tasks")
      .some((request) => request.url.includes("cursor=page-2")),
  ).toBe(true);
});
