import { act, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { MAXIMUM_PREVIEW_BYTES } from "../../api/artifacts";
import { deferred } from "../../test/deferred";
import { fakeServer } from "../../test/fake-server";
import { issueReviewFixtures } from "../../test/issue-review-fixtures";
import { renderWithAPI } from "../../test/render";
import { bytes, failure, json } from "../../test/responses";
import { FindingEvidenceFile } from "./evidence-file";
import { saveBlob } from "../../app/download";

vi.mock("../../app/download", () => ({ saveBlob: vi.fn() }));

function sample() {
  const data = issueReviewFixtures();
  const finding = data.findings.find(
    (value) => value.findingId === "issue-product-search-sql",
  )!;
  const file = data.evidence.find(
    (value) => value.metadata.artifact.name === "search-route",
  )!;
  const source = data.sources.find(
    (value) => value.receiptId === finding.firstReceiptId,
  )!;
  return { source, file };
}

it("explicitly previews and downloads exact Project evidence without an automatic payload read", async () => {
  const { source, file } = sample();
  const prefix = "/v1/projects/project_shop";
  const base = `${prefix}/artifacts/evidence/search-route`;
  const server = fakeServer({}, (_request, url) => {
    expect(url.pathname).toMatch(new RegExp(`^${base}(?:/metadata)?$`));
    expect(url.searchParams.get("revision")).toBe("search-route-r1");
    return url.pathname.endsWith("/metadata")
      ? json(file.metadata)
      : bytes(file.text, file.metadata.mediaType);
  });
  const view = renderWithAPI(
    <FindingEvidenceFile source={source} index={0} />,
    { server },
  );
  expect(server.requests).toHaveLength(0);
  await view.user.click(screen.getByRole("button", { name: "Preview" }));
  expect(
    await screen.findByText(file.text.replace(/\s+/g, " ").trim(), {
      exact: true,
    }),
  ).toBeVisible();
  await view.user.click(screen.getByRole("button", { name: "Hide preview" }));
  expect(
    screen.queryByText(file.text.replace(/\s+/g, " ").trim(), {
      exact: true,
    }),
  ).toBeNull();
  await view.user.click(
    screen.getByRole("button", { name: "Download search-route" }),
  );
  await waitFor(() => expect(saveBlob).toHaveBeenCalled());
  expect(server.sent("GET", base)).toHaveLength(2);
  expect(server.requests.every((request) => request.method === "GET")).toBe(
    true,
  );
});

it("keeps failed metadata explicit and retries without falling back to another scope", async () => {
  const { source, file } = sample();
  let failed = true;
  const server = fakeServer({}, (_request, url) => {
    expect(url.pathname).toContain("/projects/project_shop/");
    return url.pathname.endsWith("/metadata")
      ? failed
        ? failure(404, "not_found", "Evidence is unavailable")
        : json(file.metadata)
      : bytes(file.text, file.metadata.mediaType);
  });
  const view = renderWithAPI(
    <FindingEvidenceFile source={source} index={0} />,
    { server },
  );
  await view.user.click(screen.getByRole("button", { name: "Preview" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Evidence is unavailable",
  );
  expect(server.requests).toHaveLength(1);
  failed = false;
  await view.user.click(screen.getByRole("button", { name: "Retry" }));
  expect(
    await screen.findByText(file.text.replace(/\s+/g, " ").trim(), {
      exact: true,
    }),
  ).toBeVisible();
});

it("uses the shared preview size bound before reading bytes", async () => {
  const { source, file } = sample();
  const server = fakeServer({}, () =>
    json({ ...file.metadata, size: MAXIMUM_PREVIEW_BYTES + 1 }),
  );
  const view = renderWithAPI(
    <FindingEvidenceFile source={source} index={0} />,
    { server },
  );
  await view.user.click(screen.getByRole("button", { name: "Preview" }));
  expect(
    await screen.findByText(/This evidence cannot be previewed here/),
  ).toBeVisible();
  expect(server.requests).toHaveLength(1);
});

it("does not populate a changed finding's preview with a late response", async () => {
  const { source, file } = sample();
  const gate = deferred();
  const server = fakeServer({}, async (_request, url) => {
    if (url.pathname.endsWith("/metadata")) {
      await gate.promise;
      return json(file.metadata);
    }
    return bytes(file.text, file.metadata.mediaType);
  });
  const view = renderWithAPI(
    <FindingEvidenceFile source={source} index={0} />,
    { server },
  );
  await view.user.click(screen.getByRole("button", { name: "Preview" }));
  await waitFor(() => expect(server.requests).toHaveLength(1));
  view.unmount();
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  expect(server.requests).toHaveLength(1);
  expect(
    screen.queryByText(file.text.replace(/\s+/g, " ").trim(), { exact: true }),
  ).toBeNull();
});

it("shows a bounded saved source snippet by default when requested by the workspace", async () => {
  const { source, file } = sample();
  const server = fakeServer({}, (_request, url) =>
    url.pathname.endsWith("/metadata")
      ? json(file.metadata)
      : bytes(file.text, file.metadata.mediaType),
  );
  renderWithAPI(<FindingEvidenceFile source={source} index={0} snippet />, {
    server,
  });
  expect(
    await screen.findByText(file.text.replace(/\s+/g, " ").trim(), {
      exact: true,
    }),
  ).toBeVisible();
  const reads = server.requests.filter((request) =>
    new URL(request.url).pathname.includes("/artifacts/"),
  );
  expect(reads).toHaveLength(2);
  expect(
    reads.every((request) => request.url.includes("revision=search-route-r1")),
  ).toBe(true);
});
it("does not auto-read a source whose authoritative metadata exceeds the inline snippet budget", async () => {
  const { source, file } = sample();
  const server = fakeServer({}, () =>
    json({ ...file.metadata, size: 64 * 1024 + 1 }),
  );
  renderWithAPI(<FindingEvidenceFile source={source} index={0} snippet />, {
    server,
  });
  expect(
    await screen.findByRole("button", { name: "Load preview" }),
  ).toBeVisible();
  expect(server.requests).toHaveLength(1);
});
