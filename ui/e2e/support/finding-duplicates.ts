import { expect, type Page, type APIRequestContext } from "@playwright/test";
import type { components } from "../../src/api/generated/public";

type Schemas = components["schemas"];

/** Real Server/Runtime/PostgreSQL admission, followed by the existing finding UI. */
export async function exerciseFindingDuplicates(
  page: Page,
  request: APIRequestContext,
  apiURL: string,
  controlURL: string,
  token: string,
) {
  const read = async <T>(path: string): Promise<T> => {
    const response = await page.evaluate(
      async ({ apiURL, path }) => {
        const response = await fetch(`${apiURL}${path}`, {
          credentials: "include",
        });
        return { status: response.status, body: await response.json() };
      },
      { apiURL, path },
    );
    expect(response.status, path).toBe(200);
    return response.body as T;
  };
  const post = async <T>(
    path: string,
    body: unknown,
    key?: string,
  ): Promise<T> => {
    const response = await page.evaluate(
      async ({ apiURL, path, body, key }) => {
        const session = await (
          await fetch(`${apiURL}/v1/auth/session`, { credentials: "include" })
        ).json();
        const response = await fetch(`${apiURL}${path}`, {
          method: "POST",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            "X-CSRF-Token": session.csrfToken,
            ...(key === undefined ? {} : { "Idempotency-Key": key }),
          },
          body: JSON.stringify(body),
        });
        return { status: response.status, body: await response.json() };
      },
      { apiURL, path, body, key },
    );
    expect(
      [200, 201, 202],
      `${path}: ${JSON.stringify(response.body)}`,
    ).toContain(response.status);
    return response.body as T;
  };
  const project = await post<Schemas["Project"]>(
    "/v1/projects",
    {
      kind: "project",
      name: "Retained duplicate observations",
      description: "",
    },
    "correlation-project",
  );
  const fixture = await request.post(
    `${controlURL}/correlation-source/${project.projectId}`,
    { headers: { Authorization: `Bearer ${token}` } },
  );
  expect(fixture.status()).toBe(200);
  const source = (await fixture.json()) as Schemas["ExactArtifactRef"];
  const receipts: Schemas["FindingProposalReceipt"][] = [];
  for (const index of [1, 2]) {
    const run = await post<{ runId: string }>(
      `/v1/projects/${project.projectId}/runs`,
      { workflow: "source-findings-review@1", artifacts: { source } },
      `correlation-producer-${index}`,
    );
    await expect
      .poll(
        async () =>
          (await read<{ state: string }>(`/v1/runs/${run.runId}`)).state,
        { timeout: 90_000 },
      )
      .toBe("succeeded");
    const proposals = await read<Schemas["FindingProposalPage"]>(
      `/v1/runs/${run.runId}/finding-proposals`,
    );
    expect(proposals.items).toHaveLength(1);
    receipts.push(proposals.items[0]!);
  }
  expect(receipts[0]!.receiptId).not.toBe(receipts[1]!.receiptId);
  // Draft checks accept imported observations without scheduling another Worker.
  const audits: Schemas["Audit"][] = [];
  for (const index of [1, 2]) {
    audits.push(
      await post<Schemas["Audit"]>(
        `/v1/projects/${project.projectId}/audits`,
        {
          profile: { name: "owasp-top10-2025-source-risk", version: "1" },
          inputs: { source },
        },
        `correlation-audit-${index}`,
      ),
    );
  }
  const importReceipt = (
    audit: Schemas["Audit"],
    receipt: Schemas["FindingProposalReceipt"],
  ) =>
    post(`/v1/audits/${audit.auditId}/finding-proposal-imports`, {
      runId: receipt.origin.runId,
      proposal: receipt.proposal.ref,
    });
  await importReceipt(audits[0]!, receipts[0]!);
  await importReceipt(audits[1]!, receipts[0]!);
  await importReceipt(audits[1]!, receipts[1]!);
  const left = (
    await read<Schemas["AuditFindingPage"]>(
      `/v1/audits/${audits[0]!.auditId}/findings`,
    )
  ).items[0]!;
  const right = (
    await read<Schemas["AuditFindingPage"]>(
      `/v1/audits/${audits[1]!.auditId}/findings`,
    )
  ).items;
  expect(right).toHaveLength(2);
  expect(
    left.correlation?.availability,
    JSON.stringify({ anchors: left.anchors, correlation: left.correlation }),
  ).toBe("available");
  const mark = left.correlation!;
  if (mark.availability !== "available")
    throw new Error("Qualified literal did not produce a correlation");
  expect(mark.hasOtherObservations).toBe(true);
  expect(
    new Set([left, ...right].map((finding) => finding.findingId)).size,
  ).toBe(3);
  expect(
    right.every(
      (finding) =>
        finding.correlation?.availability === "available" &&
        finding.correlation.groupKey === mark.groupKey,
    ),
  ).toBe(true);
  await page.goto(`/issues/${left.auditId}/${left.findingId}`);
  const disclosure = page.locator(".issues-duplicates");
  await disclosure.locator("summary").first().click();
  await expect(disclosure.getByText(/3 retained observations/)).toBeVisible();
  await expect(disclosure.getByRole("link", { name: "Evidence" })).toHaveCount(
    3,
  );
  await expect(
    disclosure.getByText(/Same source proposal as this finding/),
  ).toHaveCount(2);
  await expect(disclosure.getByText(/Different source proposal/)).toHaveCount(
    1,
  );
  await disclosure.getByRole("link", { name: "Evidence" }).last().click();
  await expect(
    page.getByRole("heading", { name: "Embedded private key", exact: true }),
  ).toHaveText("Embedded private key");
  const retained = await read<Schemas["AuditFinding"]>(
    `/v1/audits/${right[1]!.auditId}/findings/${right[1]!.findingId}`,
  );
  expect(retained.firstProposal.evidence.length).toBeGreaterThan(0);
  expect(retained.analystVerdict).toBeUndefined();
}
