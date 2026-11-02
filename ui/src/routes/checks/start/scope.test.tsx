import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { conforming } from "../../../test/builders";
import { deferred } from "../../../test/deferred";
import { failure } from "../../../test/responses";
import { AUTOMATIC_DOCUMENT_BYTES } from "../../../app/document-limits";
import {
  materialFixture,
  sourceZip,
  trace,
  top10,
  asvsReview,
  asvsPilot,
} from "./test-fixtures";
import { renderStart } from "./test-support";
import { materialKey } from "./check-types";
import { readOpenApiEndpoints } from "../../artifacts/previews/openapi-document";
const source = JSON.stringify({
  openapi: "3.0.3",
  info: { title: "API", version: "1" },
  paths: {
    "/orders": { get: { responses: {} } },
    "/products": { post: { responses: {} } },
  },
});
const apiMaterial = materialFixture("api", "application/json", {
  size: new TextEncoder().encode(source).byteLength,
});
const profile = conforming("AuditProfile", {
  ...trace,
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
    openapi: { required: true, mediaTypes: ["application/json"] },
  },
});
const path = `/checks/new?project=project_shop&type=${profile.ref.name}`;
const content = (text: string) =>
  new Response(text, {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });

describe("selected material scope", () => {
  it("counts exact saved inline endpoints and preserves the selected revision", async () => {
    const view = renderStart(path, {
      profiles: [profile],
      materials: [sourceZip, apiMaterial],
      content: () => content(source),
    });
    expect(
      await within(
        await screen.findByRole("region", { name: "Scope" }),
      ).findByText("2 endpoints in the API spec"),
    ).toBeVisible();
    await view.user.click(screen.getByText("View endpoints"));
    expect(screen.getByText("/orders")).toBeVisible();
    const requests = view.sent(
      "GET",
      `/artifacts/${apiMaterial.artifact.namespace}/${apiMaterial.artifact.name}`,
    );
    expect(requests).toHaveLength(1);
    expect(new URL(requests[0]!.url).searchParams.get("revision")).toBe(
      apiMaterial.artifact.revision,
    );
    expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled();
  });
  it("does not claim complete coverage from local/external referenced paths or malformed content", () => {
    for (const ref of [
      "#/components/pathItems/orders",
      "https://unfetched.test/path.yaml",
    ]) {
      const parsed = readOpenApiEndpoints(
        JSON.stringify({
          openapi: "3.0.3",
          paths: { "/orders": { $ref: ref } },
        }),
      );
      expect(parsed).toEqual({ operations: [], partial: true });
    }
    for (const text of [
      "not openapi",
      "openapi: 3.0.3\npaths: []",
      "openapi: 3.0.3\npaths: {}\npaths: {}",
    ]) {
      expect(readOpenApiEndpoints(text)).toBeUndefined();
    }
  });
  it("never downloads an oversized automatic document and links to its exact preview", async () => {
    const large = { ...apiMaterial, size: AUTOMATIC_DOCUMENT_BYTES + 1 };
    const view = renderStart(path, {
      profiles: [profile],
      materials: [sourceZip, large],
    });
    expect(
      await screen.findByText(/larger than the 128 KiB automatic scope limit/),
    ).toBeVisible();
    expect(
      view.requests.filter((request) =>
        new URL(request.url).searchParams.has("revision"),
      ),
    ).toHaveLength(0);
    expect(
      screen.getByRole("link", { name: "Preview material →" }),
    ).toHaveAttribute(
      "href",
      expect.stringContaining(`revision=${large.artifact.revision}`),
    );
  });
  it("makes failed reads explicitly retryable without disabling ordinary readiness", async () => {
    let calls = 0;
    const view = renderStart(path, {
      profiles: [profile],
      materials: [sourceZip, apiMaterial],
      content: () => {
        calls++;
        return calls === 1
          ? failure(503, "storage_unavailable")
          : content(source);
      },
    });
    const scope = await screen.findByRole("region", { name: "Scope" });
    expect(
      await within(scope).findByText(/scope could not be read/),
    ).toBeVisible();
    expect(calls).toBe(1);
    expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled();
    await view.user.click(
      within(scope).getByRole("button", { name: "Try again" }),
    );
    expect(
      await within(
        await screen.findByRole("region", { name: "Scope" }),
      ).findByText("2 endpoints in the API spec"),
    ).toBeVisible();
    expect(calls).toBe(2);
  });
  it("never labels a new selection with the late result of its previous exact revision", async () => {
    const first = deferred<Response>();
    const second = materialFixture("other", "application/json", {
      size: source.length,
    });
    const view = renderStart(path, {
      profiles: [profile],
      materials: [sourceZip, apiMaterial, second],
      content: (_request, url) =>
        url.pathname.endsWith("/other") ? content(source) : first.promise,
    });
    const select = await screen.findByLabelText("Material for API spec");
    await view.user.selectOptions(select, materialKey(apiMaterial));
    await waitFor(() =>
      expect(
        view.requests.some((r) => new URL(r.url).searchParams.has("revision")),
      ).toBe(true),
    );
    await view.user.selectOptions(select, materialKey(second));
    expect(
      await within(
        await screen.findByRole("region", { name: "Scope" }),
      ).findByText("2 endpoints in the API spec"),
    ).toBeVisible();
    await act(async () => {
      first.resolve(content(source.replace('"/orders"', '"/not-current"')));
      await first.promise;
    });
    await view.user.click(screen.getByText("View endpoints"));
    expect(screen.queryByText("/not-current")).toBeNull();
  });
  it("omits source-only Scope and keeps standard variants in Advanced", async () => {
    const view = renderStart(
      `/checks/new?project=project_shop&type=${top10.ref.name}`,
      { profiles: [top10, asvsReview, asvsPilot], materials: [sourceZip] },
    );
    await screen.findByRole("button", { name: "Start check" });
    expect(screen.queryByRole("region", { name: "Scope" })).toBeNull();
    await view.user.click(screen.getByRole("link", { name: /OWASP ASVS/ }));
    await view.user.click(screen.getByText("Advanced options"));
    expect(
      screen.getByRole("group", { name: "Review coverage" }),
    ).toBeVisible();
  });
});
