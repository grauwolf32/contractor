import { webcrypto } from "node:crypto";
import { useState } from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { FindingExportRequest } from "../../../api/finding-exports";
import { queryKeys } from "../../../api/query-keys";
import { audit, finding, findingSummary } from "../../../test/builders";
import { deferred } from "../../../test/deferred";
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
    ...page(items.map(findingSummary), more ? "next" : undefined),
    auditRevision: 1,
    asOf: "2026-10-30T12:00:00Z",
    projectRevision: 1,
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
          ? failure(503, "storage_transaction_conflict", "Try again")
          : json({ ...published, replayed: true });
      },
      "GET /v1/audits/{auditId}/findings/exports/{exportId}/content": () => {
        downloadCalls++;
        return downloadCalls === 1
          ? failure(503, "storage_transaction_conflict", "Try download again")
          : bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' });
      },
    });
    const view = renderWithAPI(<AuditFindings audit={check} />, { server });
    await screen.findByText("First issue");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(screen.getByRole("button", { name: "Next" }));
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
    expect(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", { name: "Select First issue for export" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Clear selection" }),
    ).toBeDisabled();
    expect(requests).toHaveLength(1);
    const refreshed = deferred<Response>();
    server.route("GET /v1/audits/{auditId}/findings", () => refreshed.promise);
    await view.user.click(
      screen.getByRole("button", { name: "Clear selection and refresh" }),
    );
    expect(screen.getByText("0 selected for export")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Select this page" }),
    ).toBeDisabled();
    await act(() => refreshed.resolve(json(findingPage())));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Select this page" }),
      ).toBeEnabled(),
    );
    expect(requests).toHaveLength(1);
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
          items.slice(offset, offset + 50).map(findingSummary),
          offset + 50 < items.length ? String(offset + 50) : undefined,
        ),
        auditRevision: 1,
        asOf: "2026-10-30T12:00:00Z",
        projectRevision: 1,
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
    await view.user.click(screen.getByRole("button", { name: "Next" }));
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

it("keeps the selected revision when a refreshed canonical finding changes", async () => {
  let current = first;
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": () => json(findingPage([current])),
  });
  const view = renderWithAPI(<AuditFindings audit={check} />, { server });
  await screen.findByText("First issue");
  open();
  await view.user.click(
    screen.getByRole("button", { name: "Select this page" }),
  );
  current = { ...first, revision: 2 };
  await act(() =>
    view.queryClient.invalidateQueries({
      queryKey: queryKeys.audits.allFindings(check.auditId),
    }),
  );
  expect(
    await screen.findByText(/This check or a selected possible issue changed/),
  ).toBeVisible();
  expect(screen.getByText("1 selected for export")).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Create SARIF export" }),
  ).toBeDisabled();
  expect(server.sent("POST", "/exports")).toHaveLength(0);
  for (const request of server.sent("GET", "/findings"))
    expect(new URL(request.url).searchParams.get("auditRevision")).toBe(
      String(check.revision),
    );
});

for (const projection of [
  "changed finding",
  "pending read",
  "failed read",
] as const) {
  it(`replays a lost publication despite a ${projection} and downloads the saved export`, async () => {
    saved.mockClear();
    const published = await publishedExport(1, 1);
    const posts: string[] = [];
    const replay = deferred<Response>();
    const refresh = deferred<Response>();
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": () => json(findingPage([first])),
      "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
        posts.push(await request.text());
        // The first publication committed, but no response reached the client.
        if (posts.length === 1) throw new TypeError("Response lost");
        return replay.promise;
      },
      "GET /v1/audits/{auditId}/findings/exports/{exportId}/content": () =>
        bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' }),
    });
    let changeAudit: (next: typeof check) => void = () => {};
    function CurrentCheck() {
      const [current, change] = useState(check);
      changeAudit = change;
      return <AuditFindings audit={current} />;
    }
    const view = renderWithAPI(<CurrentCheck />, { server });
    await screen.findByText("First issue");
    open();
    await view.user.click(
      screen.getByRole("button", { name: "Select this page" }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText("Could not create the SARIF export");
    expect(posts).toHaveLength(1);

    if (projection === "pending read") {
      server.route("GET /v1/audits/{auditId}/findings", () =>
        refresh.promise.then((response) => response.clone()),
      );
      await act(() => changeAudit({ ...check, revision: 2 }));
      await waitFor(() =>
        expect(server.sent("GET", "/findings").length).toBeGreaterThan(1),
      );
    } else {
      server.route("GET /v1/audits/{auditId}/findings", () =>
        projection === "changed finding"
          ? json(findingPage([{ ...first, revision: 2 }]))
          : failure(503, "storage_transaction_conflict", "Read unavailable"),
      );
      await act(() =>
        view.queryClient.invalidateQueries({
          queryKey: queryKeys.audits.allFindings(check.auditId),
        }),
      );
      await screen.findByText(
        /This check or a selected possible issue changed/,
      );
    }
    expect(screen.getByText("1 selected for export")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Select this page" }),
    ).toBeDisabled();
    const retry = screen.getByRole("button", { name: "Retry export" });
    expect(retry).toBeEnabled();
    await view.user.click(retry);
    await waitFor(() => expect(posts).toHaveLength(2));
    expect(posts[1]).toBe(posts[0]);
    expect(JSON.parse(posts[1]!) as FindingExportRequest).toMatchObject({
      includeDuplicates: false,
      findings: [{ findingId: first.findingId, revision: 1 }],
    });
    expect(
      screen.getByRole("button", { name: "Creating export…" }),
    ).toBeDisabled();
    await act(() => replay.resolve(json({ ...published, replayed: true })));
    await screen.findByText(/Export ready: 1 result from 1 selected/);
    await view.user.click(
      screen.getByRole("button", { name: "Download SARIF" }),
    );
    await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
    expect(posts).toHaveLength(2);
    if (projection === "pending read")
      await act(() =>
        refresh.resolve(json({ ...findingPage([first]), auditRevision: 2 })),
      );
  });
}

it("requires clear and refresh when an unchanged ambiguous retry returns a conflict", async () => {
  const posts: string[] = [];
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": () => json(findingPage()),
    "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
      posts.push(await request.text());
      if (posts.length === 1)
        throw new TypeError("Response lost before commit");
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
  server.route("GET /v1/audits/{auditId}/findings", () =>
    json(findingPage([{ ...first, revision: 2 }, second])),
  );
  await act(() =>
    view.queryClient.invalidateQueries({
      queryKey: queryKeys.audits.allFindings(check.auditId),
    }),
  );
  await screen.findByText(/This check or a selected possible issue changed/);
  await view.user.click(screen.getByRole("button", { name: "Retry export" }));
  await waitFor(() => expect(posts).toHaveLength(2));
  expect(posts[1]).toBe(posts[0]);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Retry export" })).toBeDisabled(),
  );
  expect(screen.getByText("2 selected for export")).toBeVisible();
  expect(
    screen.getByRole("checkbox", { name: "Include duplicates" }),
  ).toBeDisabled();
  await view.user.click(
    screen.getByRole("button", { name: "Clear selection and refresh" }),
  );
  expect(screen.getByText("0 selected for export")).toBeVisible();
  expect(posts).toHaveLength(2);
});

for (const change of ["selection", "duplicate option"] as const) {
  it(`starts a new publication intent after changing the ${change} following response loss`, async () => {
    const posts: FindingExportRequest[] = [];
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": () => json(findingPage()),
      "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
        const body = (await request.json()) as FindingExportRequest;
        posts.push(body);
        if (posts.length === 1) throw new TypeError("Response lost");
        return json(
          await publishedExport(body.findings.length, body.findings.length),
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
    await screen.findByText("Could not create the SARIF export");
    await view.user.click(
      screen.getByRole("checkbox", {
        name:
          change === "selection"
            ? "Select Second issue for export"
            : "Include duplicates",
      }),
    );
    expect(
      screen.queryByRole("button", { name: "Retry export" }),
    ).not.toBeInTheDocument();
    await view.user.click(
      screen.getByRole("button", { name: "Create SARIF export" }),
    );
    await screen.findByText(/Export ready/);
    expect(posts).toHaveLength(2);
    expect(posts[1]!.clientKey).not.toBe(posts[0]!.clientKey);
    expect(posts[1]!.includeDuplicates).toBe(change === "duplicate option");
    expect(posts[1]!.findings).toHaveLength(change === "selection" ? 1 : 2);
  });
}

it("blocks publication from the prior check revision until the view refreshes", async () => {
  let revision = check.revision;
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": () =>
      json({
        ...findingPage(),
        auditRevision: revision,
      }),
  });
  let changeAudit: (next: typeof check) => void = () => {
    throw new Error("The check has not mounted");
  };
  function CurrentCheck() {
    const [current, change] = useState(check);
    changeAudit = change;
    return <AuditFindings audit={current} />;
  }
  const view = renderWithAPI(<CurrentCheck />, { server });
  await screen.findByText("First issue");
  open();
  await view.user.click(
    screen.getByRole("button", { name: "Select this page" }),
  );
  const next = { ...check, revision: check.revision + 1 };
  revision = next.revision;
  const refresh = deferred<Response>();
  // Terminal projection refresh can cancel an earlier read; every request
  // still owns a separate HTTP response body.
  server.route("GET /v1/audits/{auditId}/findings", () =>
    refresh.promise.then((response) => response.clone()),
  );
  await act(() => changeAudit(next));
  expect(
    screen.getByRole("button", { name: "Create SARIF export" }),
  ).toBeDisabled();
  await act(() =>
    refresh.resolve(json({ ...findingPage(), auditRevision: revision })),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Create SARIF export" }),
    ).toBeEnabled(),
  );
  expect(screen.getByText("2 selected for export")).toBeVisible();
});

for (const operation of ["publication", "download"] as const) {
  it(`aborts an in-flight ${operation} on leaving the check and ignores a late response`, async () => {
    saved.mockClear();
    const published = await publishedExport();
    const pending = deferred<Response>();
    let signal: AbortSignal | undefined;
    const server = fakeServer({
      "GET /v1/audits/{auditId}/findings": () => json(findingPage()),
      "POST /v1/audits/{auditId}/findings/exports": ({ request }) => {
        if (operation === "download") return json(published, 201);
        signal = request.signal;
        return pending.promise;
      },
      "GET /v1/audits/{auditId}/findings/exports/{exportId}/content": ({
        request,
      }) => {
        signal = request.signal;
        return pending.promise;
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
    if (operation === "download") {
      await screen.findByText(/Export ready/);
      await view.user.click(
        screen.getByRole("button", { name: "Download SARIF" }),
      );
    }
    await waitFor(() => expect(signal).toBeDefined());
    view.unmount();
    expect(signal!.aborted).toBe(true);
    await act(async () => {
      pending.resolve(
        operation === "publication"
          ? json(published, 201)
          : bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' }),
      );
      await pending.promise;
    });
    expect(saved).not.toHaveBeenCalled();
  });
}

it("recovers the first page after an active head advances beyond a refused revision", async () => {
  const active = audit("project", "audit", "active", { taskCount: 2 });
  let revision = active.revision;
  const fresh = deferred<Response>();
  const posts: FindingExportRequest[] = [];
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": ({ url }) => {
      const requested = Number(url.searchParams.get("auditRevision"));
      if (requested !== revision) return conflict();
      return requested === 1 ? json(findingPage()) : fresh.promise;
    },
    "POST /v1/audits/{auditId}/findings/exports": async ({ request }) => {
      posts.push((await request.json()) as FindingExportRequest);
      return json(await publishedExport(), 201);
    },
  });
  let advance: () => void = () => {};
  function CurrentCheck() {
    const [current, change] = useState(active);
    advance = () => change({ ...active, revision });
    return <AuditFindings audit={current} />;
  }
  const view = renderWithAPI(<CurrentCheck />, { server });
  await screen.findByText("First issue");
  open();
  await view.user.click(
    screen.getByRole("button", { name: "Select this page" }),
  );
  revision = 2;
  await act(() =>
    view.queryClient.invalidateQueries({
      queryKey: queryKeys.audits.allFindings(active.auditId),
    }),
  );
  await screen.findByText(/This check or a selected possible issue changed/);
  await act(advance);
  await waitFor(() =>
    expect(
      server
        .sent("GET", "/findings")
        .map((r) => new URL(r.url).searchParams.get("auditRevision")),
    ).toEqual(["1", "1", "2"]),
  );
  expect(screen.getByText("2 selected for export")).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Create SARIF export" }),
  ).toBeDisabled();
  expect(posts).toHaveLength(0);
  await act(() =>
    fresh.resolve(json({ ...findingPage(), auditRevision: revision })),
  );
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Create SARIF export" }),
    ).toBeEnabled(),
  );
  await view.user.click(
    screen.getByRole("button", { name: "Create SARIF export" }),
  );
  await screen.findByText(/Export ready/);
  expect(posts[0]!.findings).toEqual([
    { findingId: first.findingId, revision: first.revision },
    { findingId: second.findingId, revision: second.revision },
  ]);
});

it("keeps a stale continuation and selected revisions until explicit clear and refresh", async () => {
  const active = audit("project", "audit", "active", { taskCount: 2 });
  let revision = active.revision;
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": ({ url }) => {
      if (url.searchParams.has("cursor"))
        return revision === 1 ? json(findingPage([second])) : conflict();
      return json({ ...findingPage([first], true), auditRevision: revision });
    },
    "GET /v1/audits/{auditId}": () => json({ ...active, revision }),
  });
  let advance: () => void = () => {};
  function CurrentCheck() {
    const [current, change] = useState(active);
    advance = () => change({ ...active, revision });
    return <AuditFindings audit={current} />;
  }
  const view = renderWithAPI(<CurrentCheck />, { server });
  await screen.findByText("First issue");
  open();
  await view.user.click(
    screen.getByRole("button", { name: "Select this page" }),
  );
  await view.user.click(screen.getByRole("button", { name: "Next" }));
  await screen.findByText("Second issue");
  revision = 2;
  await act(advance);
  await waitFor(() => expect(server.sent("GET", "/findings")).toHaveLength(3));
  const continuation = new URL(server.sent("GET", "/findings")[2]!.url);
  expect(continuation.searchParams.get("cursor")).toBe("next");
  expect(continuation.searchParams.has("auditRevision")).toBe(false);
  await screen.findByText(/This check or a selected possible issue changed/);
  expect(screen.getByText("1 selected for export")).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Create SARIF export" }),
  ).toBeDisabled();
  expect(server.sent("POST", "/exports")).toHaveLength(0);
  await view.user.click(
    screen.getByRole("button", { name: "Clear selection and refresh" }),
  );
  await screen.findByText("First issue");
  expect(view.location().search).not.toContain("findingsCursor");
  expect(screen.getByText("0 selected for export")).toBeVisible();
});

it("keeps frozen downloads available while a newer active head read is pending", async () => {
  saved.mockClear();
  const active = audit("project", "audit", "active", { taskCount: 2 });
  const fresh = deferred<Response>();
  const server = fakeServer({
    "GET /v1/audits/{auditId}/findings": ({ url }) =>
      url.searchParams.get("auditRevision") === "1"
        ? json(findingPage())
        : fresh.promise,
    "POST /v1/audits/{auditId}/findings/exports": async () =>
      json(await publishedExport(), 201),
    "GET /v1/audits/{auditId}/findings/exports/{exportId}/content": () =>
      bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' }),
  });
  let advance: () => void = () => {};
  function CurrentCheck() {
    const [current, change] = useState(active);
    advance = () => change({ ...active, revision: 2 });
    return <AuditFindings audit={current} />;
  }
  const view = renderWithAPI(<CurrentCheck />, { server });
  await screen.findByText("First issue");
  open();
  await view.user.click(
    screen.getByRole("button", { name: "Select this page" }),
  );
  await view.user.click(
    screen.getByRole("button", { name: "Create SARIF export" }),
  );
  await screen.findByText(/Export ready/);
  await act(advance);
  await waitFor(() => expect(server.sent("GET", "/findings")).toHaveLength(2));
  await view.user.click(screen.getByRole("button", { name: "Download SARIF" }));
  await waitFor(() => expect(saved).toHaveBeenCalledTimes(1));
  expect(server.sent("POST", "/exports")).toHaveLength(1);
  await act(() => fresh.resolve(json({ ...findingPage(), auditRevision: 2 })));
});
