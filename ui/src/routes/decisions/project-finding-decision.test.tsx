import { useQuery } from "@tanstack/react-query";
import { screen, waitFor, within } from "@testing-library/react";
import { expect, it } from "vitest";
import { getProjectFinding, type ProjectFinding } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { conforming, DIGEST, NOW } from "../../test/builders";
import { deferred } from "../../test/deferred";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { failure, json, preconditionFailed } from "../../test/responses";
import { FindingDecision } from "./finding-decision";

function sample(): ProjectFinding {
  return conforming("ProjectFinding", {
    findingId: "finding_shared",
    projectId: "project_shared",
    firstReceiptId: "receipt_a",
    firstProposal: {
      ref: { namespace: "finding-materials", name: "claim", revision: "r1" },
      digest: DIGEST,
      mediaType: "application/json",
      sizeBytes: 200,
    },
    title: "Missing ownership check",
    claim: {
      title: "Missing ownership check",
      description: "A claimed authorization problem.",
      subject: null,
      preconditions: [],
      standard_refs: [],
      severity_suggestion: "high",
      limitations: [],
    },
    state: null,
    analystVerdict: null,
    analystSeverity: null,
    rationale: null,
    actorId: null,
    assessedAt: null,
    duplicateTargetId: null,
    revision: 1,
    createdAt: NOW,
    updatedAt: NOW,
  });
}
function Workspace() {
  const api = usePublicAPI();
  const project = useQuery({
    queryKey: ["project-findings", "project_shared", "finding_shared"],
    queryFn: () => getProjectFinding(api, "project_shared", "finding_shared"),
  });
  const left = useQuery({
    queryKey: ["audits", "left", "finding"],
    queryFn: () => getProjectFinding(api, "project_shared", "finding_shared"),
  });
  const right = useQuery({
    queryKey: ["audits", "right", "finding"],
    queryFn: () => getProjectFinding(api, "project_shared", "finding_shared"),
  });
  return (
    <>
      {(
        [
          ["Project", project],
          ["First check", left],
          ["Second check", right],
        ] as const
      ).map(([name, q]) => (
        <p key={String(name)} aria-label={String(name)}>
          {typeof q === "string"
            ? null
            : q.data === undefined
              ? "Loading…"
              : `${q.data.state ?? "Not assessed"} · ${q.data.revision}`}
        </p>
      ))}
      {left.data === undefined ? null : (
        <FindingDecision presentation="review" finding={left.data} />
      )}
    </>
  );
}
function reads(value: () => ProjectFinding) {
  return {
    "GET /v1/projects/{projectId}/findings/{findingId}": () =>
      json(value(), 200, { ETag: `"${value().revision}"` }),
    "GET /v1/audits/{auditId}/findings/{findingId}": () =>
      json(value(), 200, { ETag: `"${value().revision}"` }),
  };
}
it("edits the Project directly from a check and refetches the same state in both checks", async () => {
  let current = sample();
  const writes: { headers: Headers; body: unknown }[] = [];
  const gate = deferred<void>();
  const refreshGate = deferred<void>();
  let refreshReads = 0;
  const server = fakeServer({
    ...reads(() => current),
    "GET /v1/projects/{projectId}/findings/{findingId}": async () => {
      if (current.revision === 2 && ++refreshReads === 3)
        await refreshGate.promise;
      return json(current, 200, { ETag: `"${current.revision}"` });
    },
    "PATCH /v1/projects/{projectId}/findings/{findingId}": async ({
      request,
    }) => {
      const body = await request.json();
      writes.push({ headers: request.headers, body });
      await gate.promise;
      current = {
        ...current,
        revision: 2,
        state: "confirmed",
        analystVerdict: "true_positive",
        analystSeverity: "high",
        rationale: body.rationale,
        actorId: "analyst",
        assessedAt: NOW,
      };
      return json({ finding: current, replayed: false }, 200, { ETag: '"2"' });
    },
  });
  const view = renderWithAPI(<Workspace />, { server });
  await screen.findByRole("button", { name: /Confirm issue/ });
  expect(screen.queryByRole("textbox", { name: /Why/ })).toBeNull();
  expect(
    screen.queryByRole("button", { name: "Use issue summary" }),
  ).toBeNull();
  expect(screen.queryByText("Choose a decision.")).toBeNull();
  await view.user.click(screen.getByRole("button", { name: /Confirm issue/ }));
  await view.user.click(screen.getByRole("radio", { name: "High" }));
  await view.user.type(
    screen.getByRole("textbox", { name: /Why/ }),
    "Verified the ownership predicate.",
  );
  await view.user.click(
    screen.getByRole("button", { name: "Record decision" }),
  );
  await waitFor(() => expect(writes).toHaveLength(1));
  expect(writes[0]!.headers.get("If-Match")).toBe('"1"');
  expect(writes[0]!.headers.get("Idempotency-Key")).toBeTruthy();
  expect(writes[0]!.body).toEqual({
    verdict: "true_positive",
    severity: "high",
    rationale: "Verified the ownership predicate.",
  });
  expect(
    screen.getByRole("button", { name: "Record decision" }),
  ).toHaveAttribute("aria-disabled", "true");
  gate.resolve();
  await waitFor(() =>
    expect(screen.getByLabelText("First check")).toHaveTextContent(
      "confirmed · 2",
    ),
  );
  expect(
    screen.getByRole("button", { name: "Record decision" }),
  ).toHaveAttribute("aria-disabled", "true");
  expect(screen.queryByRole("button", { name: "Change decision" })).toBeNull();
  refreshGate.resolve();
  for (const name of ["Project", "First check", "Second check"])
    await waitFor(() =>
      expect(screen.getByLabelText(name)).toHaveTextContent("confirmed · 2"),
    );
  await view.user.click(
    await screen.findByRole("button", { name: "Change decision" }),
  );
  expect(screen.getByRole("button", { name: "Confirm issue" })).toBeVisible();
  expect(screen.queryByRole("textbox", { name: /Why/ })).toBeNull();
  expect(
    server.requests.every((request) => !request.url.includes("/reviews")),
  ).toBe(true);
});
it("preserves the draft after a stale edit and requires the current revision", async () => {
  let current = sample();
  let attempts = 0;
  const server = fakeServer({
    ...reads(() => current),
    "PATCH /v1/projects/{projectId}/findings/{findingId}": () => {
      attempts++;
      current = {
        ...current,
        revision: 9,
        state: "rejected",
        analystVerdict: "false_positive",
        rationale: "Another analyst",
        actorId: "colleague",
        assessedAt: NOW,
      };
      return preconditionFailed();
    },
  });
  const view = renderWithAPI(<Workspace />, { server });
  await view.user.click(
    await screen.findByRole("button", { name: /Needs evidence/ }),
  );
  await view.user.type(
    screen.getByRole("textbox", { name: /Why/ }),
    "The controlled check is still missing.",
  );
  await view.user.click(
    screen.getByRole("button", { name: "Record decision" }),
  );
  await screen.findByText(/This possible issue changed/);
  expect(screen.getByRole("textbox", { name: /Why/ })).toHaveValue(
    "The controlled check is still missing.",
  );
  expect(
    screen.getByRole("button", { name: "Record decision" }),
  ).toBeDisabled();
  expect(attempts).toBe(1);
  await view.user.click(
    screen.getByRole("button", { name: "Use current revision" }),
  );
  expect(screen.getByRole("textbox", { name: /Why/ })).toHaveValue(
    "The controlled check is still missing.",
  );
});
it("retries an ambiguous save with its original body, revision and idempotency key", async () => {
  let current = sample();
  const attempts: {
    key: string | null;
    match: string | null;
    body: unknown;
  }[] = [];
  const server = fakeServer({
    ...reads(() => current),
    "PATCH /v1/projects/{projectId}/findings/{findingId}": async ({
      request,
    }) => {
      const body = await request.json();
      attempts.push({
        key: request.headers.get("Idempotency-Key"),
        match: request.headers.get("If-Match"),
        body,
      });
      if (attempts.length === 1)
        return failure(
          503,
          "storage_transaction_conflict",
          "Storage retry required",
        );
      current = {
        ...current,
        revision: 2,
        state: "needs-evidence",
        analystVerdict: "needs_evidence",
        rationale: body.rationale,
        actorId: "analyst",
        assessedAt: NOW,
      };
      return json({ finding: current, replayed: true }, 200, { ETag: '"2"' });
    },
  });
  const view = renderWithAPI(<Workspace />, { server });
  await view.user.click(
    await screen.findByRole("button", { name: /Needs evidence/ }),
  );
  await view.user.type(
    screen.getByRole("textbox", { name: /Why/ }),
    "Need the exact response.",
  );
  await view.user.click(
    screen.getByRole("button", { name: "Record decision" }),
  );
  await view.user.click(
    await screen.findByRole("button", { name: "Retry this save" }),
  );
  await waitFor(() =>
    expect(screen.getByLabelText("Project")).toHaveTextContent(
      "needs-evidence · 2",
    ),
  );
  expect(attempts).toHaveLength(2);
  expect(attempts[1]).toEqual(attempts[0]);
  expect(
    within(
      screen.getByRole("region", { name: "Assess possible issue" }),
    ).getByRole("status"),
  ).toHaveTextContent("Assessment saved");
});
