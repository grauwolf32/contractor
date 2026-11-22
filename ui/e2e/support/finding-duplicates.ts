import { expect, type Page, type APIRequestContext } from "@playwright/test";
import type { components } from "../../src/api/generated/public";
import { exerciseFindingExports } from "./finding-exports";

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
      { workflow: "source-findings-review@4", artifacts: { source } },
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
          profile: { name: "owasp-top10-2025-source-risk", version: "7" },
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

  expect(right).toHaveLength(1);
  expect(right[0]!.findingId).toBe(left.findingId);
  expect(left.state).toBeNull();
  expect(right[0]!.state).toBeNull();
  const sources = await read<Schemas["ProjectFindingSourcePage"]>(
    `/v1/projects/${project.projectId}/finding-sources?findingId=${left.findingId}`,
  );
  expect(sources.items.map((source) => source.receiptId).sort()).toEqual(
    receipts.map((source) => source.receiptId).sort(),
  );
  await page.goto(`/issues/${project.projectId}/${left.findingId}`);
  await expect(
    page.getByRole("heading", { name: "Embedded private key", exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "History" }).click();
  const independent = page.getByRole("region", { name: "Independent sources" });
  await expect(independent.getByRole("button", { pressed: true })).toHaveCount(
    1,
  );
  const retained = await read<Schemas["FindingProposalReceipt"]>(
    `/v1/projects/${project.projectId}/finding-sources/${receipts[1]!.receiptId}`,
  );
  expect(retained.projectEvidence!.length).toBeGreaterThan(0);
  await exerciseFindingExports(page, audits[1]!, right);
}
