import { webcrypto } from "node:crypto";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { failure, json } from "../../../test/responses";
import { draftAudit, projectFixture, webScanProfile } from "./test-fixtures";
import { renderStart } from "./test-support";
import settingsCases from "../../../../../api/testdata/audit-web-scan-settings.json";
import { conformsTo } from "../../../api/contract";

beforeAll(() => vi.stubGlobal("crypto", webcrypto));
afterAll(() => vi.unstubAllGlobals());

const start = "/checks/new?project=project_shop&type=web-scan";
const project = projectFixture({ httpTarget: { url: "https://shop.test/" } });

describe("Web scan Audit", () => {
  it("agrees with the Server on the shared Nuclei selection fixtures", () => {
    expect(settingsCases.length).toBeGreaterThan(0);
    for (const test of settingsCases)
      expect(conformsTo("AuditWebScanSettings", test.settings), test.name).toBe(
        test.valid,
      );
  });
  it("starts a real Audit from a URL without uploaded OpenAPI or source", async () => {
    const view = renderStart(start, { profiles: [webScanProfile], project });
    expect(await screen.findByLabelText(/Target URL/)).toHaveValue(
      "https://shop.test/",
    );
    expect(
      screen.queryByRole("combobox", { name: /Project scope/ }),
    ).toBeNull();
    await view.user.click(screen.getByLabelText(/Target URL/));
    await view.user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
    const request = await view.sent("POST", "/audits")[0]!.json();
    expect(request.profile).toEqual({ name: "web-scan", version: "1" });
    expect(Object.keys(request.inputs).sort()).toEqual(["settings", "target"]);
    expect(request.scope).toEqual({ target: "https://shop.test/" });
    await waitFor(() => expect(view.sent("POST", "/start")).toHaveLength(1));
    expect(view.sent("POST", "/runs")).toHaveLength(0);
    await waitFor(() =>
      expect(view.router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
  });

  it("requires a custom rule and saves the exact uploaded YAML in this project", async () => {
    const view = renderStart(start, { profiles: [webScanProfile], project });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(
      screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
    );
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    await view.user.upload(
      screen.getByLabelText("Upload Nuclei rules"),
      new File(["id: project-only-rule\n"], "my-rule.yaml", {
        type: "application/yaml",
      }),
    );
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
    const request = await view.sent("POST", "/audits")[0]!.json();
    expect(request.inputs.rulepack.namespace).toBe("scan-inputs");
    const writes = view.requests.filter((request) => request.method === "PUT");
    expect(writes).toHaveLength(3);
    expect(
      writes.every((request) =>
        new URL(request.url).pathname.startsWith(
          "/v1/projects/project_shop/artifacts/",
        ),
      ),
    ).toBe(true);
    expect(await writes[2]!.text()).toBe("id: project-only-rule\n");
    expect(view.sent("POST", "/scan-sweeps")).toHaveLength(0);
  });

  it("reuses retained inputs and create keys after an ambiguous Audit creation", async () => {
    let attempts = 0;
    const view = renderStart(start, {
      profiles: [webScanProfile],
      project,
      create: () => {
        attempts++;
        if (attempts === 1) throw new TypeError("Lost response");
        return json(draftAudit(webScanProfile), 201, { ETag: '"1"' });
      },
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    await screen.findByText("The check may already exist.");
    await view.user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(2));
    const calls = view.sent("POST", "/audits");
    expect(calls[0]!.headers.get("Idempotency-Key")).toBe(
      calls[1]!.headers.get("Idempotency-Key"),
    );
    expect(
      view.requests.filter((request) => request.method === "PUT"),
    ).toHaveLength(2);
  });

  it("keeps upload failures visible and never creates an Audit without its inputs", async () => {
    const view = renderStart(start, {
      profiles: [webScanProfile],
      project,
      upload: () =>
        failure(503, "storage_transaction_conflict", "Storage is busy"),
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    expect(
      await screen.findByText(/Could not save the scan inputs/),
    ).toBeVisible();
    expect(view.sent("POST", "/audits")).toHaveLength(0);
  });

  it("treats different rule bytes as a new selection even when file metadata matches", async () => {
    const view = renderStart(start, {
      profiles: [webScanProfile],
      project,
      create: () => failure(400, "invalid_request", "Rejected by the fixture"),
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(
      screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
    );
    for (const text of ["id: one\n", "id: two\n"]) {
      await view.user.upload(
        screen.getByLabelText("Upload Nuclei rules"),
        new File([text], "same.yaml", {
          type: "application/yaml",
          lastModified: 1,
        }),
      );
      await view.user.click(
        screen.getByRole("button", { name: "Start check" }),
      );
      await screen.findByText("The check was not created.");
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Start check" }),
        ).toBeEnabled(),
      );
    }
    const requests = view.sent("POST", "/audits");
    expect(requests).toHaveLength(2);
    const first = await requests[0]!.json();
    const second = await requests[1]!.json();
    expect(first.inputs.rulepack).not.toEqual(second.inputs.rulepack);
  });
});
