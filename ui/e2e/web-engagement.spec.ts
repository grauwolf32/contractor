import { randomUUID } from "node:crypto";
import { expect, test } from "@playwright/test";

test.skip(
  process.env.CONTRACTOR_SCAN_BROWSER !== "1",
  "Requires the real scanner stack",
);

test("browser launches a web Audit through engagement without a settings artifact", async ({
  page,
}) => {
  const api = process.env.CONTRACTOR_UI_E2E_API_URL!;
  const origin = process.env.CONTRACTOR_UI_E2E_BASE_URL!;
  const target = process.env.CONTRACTOR_SCAN_TARGET_URL!;
  await page.goto("/projects");
  await page
    .getByLabel("Username")
    .fill(process.env.CONTRACTOR_UI_E2E_USERNAME!);
  await page
    .getByLabel("Password")
    .fill(process.env.CONTRACTOR_UI_E2E_PASSWORD!);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Account", exact: true }),
  ).toBeVisible();
  const session = await page.request.get(api + "/v1/auth/session");
  const { csrfToken } = (await session.json()) as { csrfToken: string };
  const headers = () => ({
    "X-CSRF-Token": csrfToken,
    Origin: origin,
    "Idempotency-Key": randomUUID(),
  });
  const created = await page.request.post(api + "/v1/projects", {
    headers: headers(),
    data: {
      kind: "project",
      name: "Browser engagement scan",
      activeChecks: "automatic",
    },
  });
  expect(created.status()).toBe(201);
  const { projectId } = (await created.json()) as { projectId: string };
  const address = new URL(target);
  const policy = await page.request.put(
    api + `/v1/projects/${projectId}/artifacts/fixture/engagement`,
    {
      headers: {
        ...headers(),
        "If-None-Match": "*",
        "Content-Type": "application/vnd.contractor.engagement+json",
      },
      data: Buffer.from(
        JSON.stringify({
          schema: "contractor.engagement.v1",
          egress: {
            routes: [
              {
                origin: address.origin,
                methods: ["GET"],
                pathPrefixes: ["/"],
                addressRanges: [address.hostname + "/32"],
              },
            ],
            exclusions: [],
          },
          identities: [
            { id: "anonymous", kind: "anonymous", origins: [address.origin] },
          ],
        }),
      ),
    },
  );
  expect(policy.status()).toBe(201);
  const writes: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "PUT") writes.push(request.url());
  });
  await page.goto(`/checks/new?project=${projectId}&type=web-scan`);
  await page.getByLabel("Target URL", { exact: false }).fill(target);
  const creation = page.waitForResponse(
    (response) =>
      response.url() === api + `/v1/projects/${projectId}/audits` &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start check", exact: true }).click();
  const response = await creation;
  expect(response.status()).toBe(201);
  const audit = (await response.json()) as {
    auditId: string;
    inputs: Record<string, unknown>;
  };
  expect(Object.keys(audit.inputs).sort()).toEqual(["engagement", "target"]);
  expect(writes).toHaveLength(1);
  await expect
    .poll(
      async () => {
        const read = await page.request.get(
          api + `/v1/audits/${audit.auditId}`,
        );
        return ((await read.json()) as { state: string }).state;
      },
      { timeout: 120000 },
    )
    .toBe("completed");
  const taskResponse = await page.request.get(
    api + `/v1/audits/${audit.auditId}/tasks`,
  );
  const { items } = (await taskResponse.json()) as {
    items: { gap_code?: string; findings?: unknown }[];
  };
  expect(items).toHaveLength(1);
  expect(items[0]?.gap_code).toBeUndefined();
  expect(items[0]?.findings).toBeDefined();
});
