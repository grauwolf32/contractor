import { webcrypto } from "node:crypto";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { FindingExportRequest } from "../../../api/finding-exports";
import { audit, finding } from "../../../test/builders";
import { fakeServer } from "../../../test/fake-server";
import { EXPORT_BYTES, publishedExport } from "../../../test/finding-export";
import { renderWithAPI } from "../../../test/render";
import { bytes, conflict, failure, json, page } from "../../../test/responses";
import { AuditFindings } from "./audit-findings";

const saved = vi.hoisted(() => vi.fn());
vi.mock("../../../app/download", () => ({ saveBlob: saved }));
beforeAll(() => vi.stubGlobal("crypto", webcrypto));
const check = audit("project", "audit", "completed");
const first = finding("audit", "one", {}, { title: "First issue" });
const second = finding("audit", "two", {}, { title: "Second issue" });
function open() {
  const details = screen
    .getByText("Export SARIF", { selector: "summary" })
    .closest("details")!;
  details.open = true;
  fireEvent(details, new Event("toggle", { bubbles: true }));
}
function findingPage(items = [first, second], more = false) {
  return {
    ...page(items, more ? "next" : undefined),
    auditRevision: 1,
    asOf: "2026-10-30T12:00:00Z",
    total: 2,
  };
}

describe("check SARIF export", () => {
  it("preserves selections across pages, defaults to no duplicates and retries a lost publication with the same key", async () => {
    const published = await publishedExport();
    const posts: FindingExportRequest[] = [];
    let downloadCalls = 0;
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": ({ url }) =>
        json(
          url.searchParams.has("cursor")
            ? findingPage([second])
            : findingPage([first], true),
        ),
      "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
        posts.push((await request.json()) as FindingExportRequest);
        return posts.length === 1
          ? failure(503, "storage_transaction_conflict", "Try again", true)
          : json({ ...published, replayed: true });
      },
      "GET /v1/audits/{auditId}/findings/exports/{exportId}/content": () => {
        downloadCalls++;
        return downloadCalls === 1
          ? failure(
              503,
              "storage_transaction_conflict",
              "Try download again",
              true,
            )
          : bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' });
      },
    });
    const view = renderWithAPI(<AuditFindings audit={check} />, { server });
    await screen.findByText("First issue");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(screen.getByRole("button", { name: "Next page" }));
    await screen.findByText("Second issue");
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    expect(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    ).not.toBeChecked();
    expect(screen.getByText("2 selected for export")).toBeVisible();
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText("Could not create the SARIF export");
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(screen.getByRole("button", { name: "Retry export" }));
    await screen.findByText(/Export ready: 1 result from 2 selected/);
    expect(posts).toHaveLength(2);
    expect(posts[1]).toEqual(posts[0]);
    expect(posts[0]).toMatchObject({
      includeDuplicates: false,
      findings: [
        { findingId: "one", revision: 1 },
        { findingId: "two", revision: 1 },
      ],
    });
    await view.user.click(
      screen.getByRole("button", { name: "Download SARIF" }),
    );
    await screen.findByText("Could not download the saved export");
    await view.user.click(
      screen.getByRole("button", { name: "Download SARIF" }),
    );
    await waitFor(() => expect(saved).toHaveBeenCalled());
    expect(posts).toHaveLength(2);
    expect(downloadCalls).toBe(2);
  });
  it("changes publication identity with the duplicate option and reports the server's expanded count", async () => {
    const requests: FindingExportRequest[] = [];
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": () => json(findingPage()),
      "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
        const value = (await request.json()) as FindingExportRequest;
        requests.push(value);
        return json(
          await publishedExport(2, value.includeDuplicates ? 2 : 1),
          201,
        );
      },
    });
    const view = renderWithAPI(<AuditFindings audit={check} />, { server });
    await screen.findByText("First issue");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText(/Export ready: 1 result/);
    await view.user.click(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText(/Export ready: 2 results/);
    expect(requests[0]!.clientKey).not.toBe(requests[1]!.clientKey);
    expect(requests[1]!.includeDuplicates).toBe(true);
  });
  it("keeps stale revisions on conflict and requires an explicit clear/refresh", async () => {
    const requests: FindingExportRequest[] = [];
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": () => json(findingPage()),
      "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
        requests.push((await request.json()) as FindingExportRequest);
        return conflict();
      },
    });
    const view = renderWithAPI(<AuditFindings audit={check} />, { server });
    await screen.findByText("First issue");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText("Could not create the SARIF export");
    expect(screen.getByRole("button", { name: "Retry export" })).toBeDisabled();
    expect(requests).toHaveLength(1);
    await view.user.click(
      screen.getByRole("button", { name: "Clear selection and refresh" }),
    );
    expect(screen.getByText("0 selected for export")).toBeVisible();
  });
  it("resets selection and the option when switching to another check", async () => {
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": ({ params }) =>
        json(findingPage([finding(params.auditId!, "one")])),
    });
    const view = renderWithAPI(<AuditFindings audit={check} />, {
      server,
      routes: [
        { path: "/", element: <AuditFindings audit={check} /> },
        {
          path: "/other",
          element: (
            <AuditFindings audit={audit("project", "other", "completed")} />
          ),
        },
      ],
    });
    await screen.findByText("Missing object authorization");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    );
    await act(() => view.router.navigate("/other"));
    await screen.findByText("Missing object authorization");
    open();
    expect(screen.getByText("0 selected for export")).toBeVisible();
    expect(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    ).not.toBeChecked();
  });
});

it("bounds a multi-page selection at 256 without selecting a hidden remainder", async () => {
  const items = Array.from({ length: 257 }, (_, index) =>
    finding("audit", `finding-${index}`, {}, { title: `Issue ${index}` }),
  );
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": ({ url }) => {
      const offset = Number(url.searchParams.get("cursor") ?? "0");
      return json({
        ...page(
          items.slice(offset, offset + 50),
          offset + 50 < items.length ? String(offset + 50) : undefined,
        ),
        auditRevision: 1,
        asOf: "2026-10-30T12:00:00Z",
        total: items.length,
      });
    },
  });
  const view = renderWithAPI(<AuditFindings audit={check} />, { server });
  await screen.findByText("Issue 0");
  open();
  for (let offset = 0; offset < 250; offset += 50) {
    await screen.findByText(`Issue ${offset}`);
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(screen.getByRole("button", { name: "Next page" }));
  }
  await screen.findByText("Issue 250");
  expect(
    screen.getByRole("button", { name: "Select this page" }),
  ).toBeDisabled();
  for (let index = 250; index < 256; index++)
    await view.user.click(
      screen.getByRole("checkbox", {
        name: `Select Issue ${index} for export`,
      }),
    );
  expect(screen.getByText("256 selected for export")).toBeVisible();
  expect(
    screen.getByRole("checkbox", { name: "Select Issue 256 for export" }),
  ).toBeDisabled();
  expect(server.sent("POST", "/exports")).toHaveLength(0);
});
