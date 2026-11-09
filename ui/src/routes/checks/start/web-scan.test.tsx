import { webcrypto } from "node:crypto";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { act, screen, waitFor } from "@testing-library/react";
import { failure, json, page } from "../../../test/responses";
import {
  draftAudit,
  projectFixture,
  startResponse,
  top10,
  sourceZip,
  webScanProfile,
  engagementMaterial,
} from "./test-fixtures";
import {
  renderStart as renderBaseStart,
  type FakeStartServer,
} from "./test-support";
import { renderWithAPI } from "../../../test/render";
import { useWebScanDraft, type WebScanDraft } from "./web-scan-state";
import selectionCases from "../../../../../api/testdata/audit-web-scan-parameters.json";
import { conformsTo } from "../../../api/contract";
import { deferred } from "../../../test/deferred";
import { artifactMetadata, artifactWrite } from "../../../test/builders";

function renderStart(path: string, server: FakeStartServer = {}) {
  return renderBaseStart(path, {
    ...server,
    materials:
      typeof server.materials === "function"
        ? server.materials
        : [engagementMaterial, ...(server.materials ?? [])],
  });
}

beforeAll(() => vi.stubGlobal("crypto", webcrypto));
afterAll(() => vi.unstubAllGlobals());

const start = "/checks/new?project=project_shop&type=web-scan";
const project = projectFixture({ httpTarget: { url: "https://shop.test/" } });

describe("Web scan Audit", () => {
  it("requires an engagement before preparing target artifacts", async () => {
    const view = renderBaseStart(start, {
      profiles: [webScanProfile],
      project,
      materials: [],
    });
    await screen.findByLabelText(/Target URL/);
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    expect(
      view.requests.filter((r) => r.method === "PUT" || r.method === "POST"),
    ).toHaveLength(0);
  });
  it.each([
    { name: "rules.yaml", size: 1048577, limit: "1 MiB" },
    { name: "rules.zip", size: 16777217, limit: "16 MiB" },
  ])(
    "refuses oversized $name before reading or preparing inputs",
    async ({ name, size, limit }) => {
      const view = renderStart(start, { profiles: [webScanProfile], project });
      await screen.findByLabelText(/Target URL/);
      await view.user.click(
        screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
      );
      const file = new File(["x"], name);
      const read = vi.fn(async () => new ArrayBuffer(1));
      Object.defineProperties(file, {
        size: { value: size },
        arrayBuffer: { value: read },
      });
      const hash = vi.spyOn(crypto.subtle, "digest");
      try {
        await view.user.upload(
          screen.getByLabelText("Upload Nuclei rules"),
          file,
        );
        expect(
          screen.getByRole("button", { name: "Start check" }),
        ).toBeDisabled();
        expect(
          screen.getByRole("button", { name: "Save as draft" }),
        ).toBeDisabled();
        expect(
          screen.getByText(new RegExp(`at most ${limit}`)),
        ).toBeInTheDocument();
        await view.user.click(screen.getByLabelText(/Target URL/));
        await view.user.keyboard("{Control>}{Enter}{/Control}");
        expect(read).not.toHaveBeenCalled();
        expect(hash).not.toHaveBeenCalled();
        expect(
          view.requests.filter(
            (request) => request.method === "PUT" || request.method === "POST",
          ),
        ).toHaveLength(0);
      } finally {
        hash.mockRestore();
      }
    },
  );

  it.each([
    { mediaType: "application/yaml", size: 1048577, limit: "1 MiB" },
    {
      mediaType: "application/vnd.contractor.nuclei-rulepack+zip",
      size: 16777217,
      limit: "16 MiB",
    },
  ])(
    "refuses oversized selected $mediaType metadata",
    async ({ mediaType, size, limit }) => {
      const pack = artifactMetadata("rules", "oversized", { mediaType, size });
      const view = renderStart(start, {
        profiles: [webScanProfile],
        project,
        materials: [pack],
      });
      await screen.findByLabelText(/Target URL/);
      await view.user.click(
        screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
      );
      await view.user.selectOptions(
        screen.getByLabelText("Project rulepack"),
        `${pack.artifact.namespace}/${pack.artifact.name}@${pack.artifact.revision}`,
      );
      expect(
        screen.getByRole("button", { name: "Start check" }),
      ).toBeDisabled();
      expect(
        screen.getByRole("button", { name: "Save as draft" }),
      ).toBeDisabled();
      expect(
        screen.getByText(new RegExp(`at most ${limit}`)),
      ).toBeInTheDocument();
      expect(
        view.requests.filter(
          (request) => request.method === "PUT" || request.method === "POST",
        ),
      ).toHaveLength(0);
    },
  );

  it.each([
    { name: "rules.yaml", mediaType: "application/yaml", size: 1048576 },
    {
      name: "rules.zip",
      mediaType: "application/vnd.contractor.nuclei-rulepack+zip",
      size: 16777216,
    },
  ])(
    "keeps boundary-sized $name selectable from file and later material pages",
    async ({ name, mediaType, size }) => {
      const pack = artifactMetadata("rules", "boundary", { mediaType, size });
      const tooBig = artifactMetadata("rules", "later-oversized", {
        mediaType,
        size: size + 1,
      });
      const view = renderStart(start, {
        profiles: [webScanProfile],
        project,
        materials: (_request, url) => {
          const index = Number(url.searchParams.get("cursor") ?? "0");
          return json(
            page(
              index === 4
                ? [tooBig]
                : index === 0
                  ? [pack, engagementMaterial]
                  : [
                      artifactMetadata("other", `page-${index}`, {
                        mediaType: "text/plain",
                      }),
                    ],
              index < 4 ? String(index + 1) : undefined,
            ),
          );
        },
      });
      await screen.findByLabelText(/Target URL/);
      await view.user.click(
        screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
      );
      const file = new File(["x"], name);
      Object.defineProperty(file, "size", { value: size });
      await view.user.upload(
        screen.getByLabelText("Upload Nuclei rules"),
        file,
      );
      await view.user.selectOptions(
        screen.getByLabelText("Material for Engagement"),
        JSON.stringify([
          engagementMaterial.artifact.namespace,
          engagementMaterial.artifact.name,
          engagementMaterial.artifact.revision,
        ]),
      );
      expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled();
      expect(
        screen.getByRole("button", { name: "Save as draft" }),
      ).toBeEnabled();
      await view.user.selectOptions(
        screen.getByLabelText("Project rulepack"),
        `${pack.artifact.namespace}/${pack.artifact.name}@${pack.artifact.revision}`,
      );
      expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled();
      await view.user.click(
        screen.getByRole("button", { name: "Load more project rules" }),
      );
      const key = `${tooBig.artifact.namespace}/${tooBig.artifact.name}@${tooBig.artifact.revision}`;
      await screen.findByRole("option", {
        name: new RegExp("later-oversized"),
      });
      await view.user.selectOptions(
        screen.getByLabelText("Project rulepack"),
        key,
      );
      expect(
        screen.getByRole("button", { name: "Start check" }),
      ).toBeDisabled();
      expect(
        screen.getByRole("button", { name: "Save as draft" }),
      ).toBeDisabled();
      expect(
        view.requests.filter(
          (request) => request.method === "PUT" || request.method === "POST",
        ),
      ).toHaveLength(0);
    },
  );

  it("guards direct preparation before target writes and file reads", async () => {
    let draft: WebScanDraft | undefined;
    function Probe() {
      draft = useWebScanDraft(project.projectId, project, webScanProfile);
      return null;
    }
    const view = renderWithAPI(<Probe />);
    const file = new File(["x"], "rules.yaml");
    const read = vi.fn(async () => new ArrayBuffer(1));
    Object.defineProperties(file, {
      size: { value: 1048577 },
      arrayBuffer: { value: read },
    });
    act(() => {
      draft!.setCustom(true);
      draft!.setFile(file);
    });
    await expect(draft!.prepare(() => {})).rejects.toThrow(/at most 1 MiB/);
    expect(read).not.toHaveBeenCalled();
    expect(view.requests).toHaveLength(0);
  });

  it.each([false, true])(
    "retains shared launch options for custom=%s",
    async (custom) => {
      const view = renderStart(start, { profiles: [webScanProfile], project });
      await screen.findByLabelText(/Target URL/);
      if (custom) {
        await view.user.click(
          screen.getByRole("radio", { name: /Custom Nuclei rules/ }),
        );
        await view.user.upload(
          screen.getByLabelText("Upload Nuclei rules"),
          new File(["id: fixture\n"], "rule.yaml"),
        );
      }
      await view.user.click(screen.getByText("Advanced options"));
      await view.user.click(screen.getByRole("button", { name: "+ cloud" }));
      await view.user.type(screen.getByLabelText("Label value 1"), "aws");
      await view.user.click(screen.getByText("For admins and debugging."));
      await view.user.type(
        screen.getByLabelText("Runtime configuration labels"),
        "debug, caido",
      );
      await view.user.selectOptions(
        screen.getByLabelText("Time limit"),
        "604800",
      );
      await view.user.click(
        screen.getByRole("button", { name: "Start check" }),
      );
      await waitFor(() => expect(view.sent("POST", "/start")).toHaveLength(1));
      expect(await view.sent("POST", "/audits")[0]!.json()).toMatchObject({
        labels: { cloud: "aws" },
        runtimeLabels: ["debug", "caido"],
        scope: { target: "https://shop.test/" },
      });
      expect(await view.sent("POST", "/start")[0]!.json()).toMatchObject({
        deadlineSeconds: 604800,
      });
    },
  );

  it("validates shared labels and deadline before preparing scanner inputs", async () => {
    const view = renderStart(start, { profiles: [webScanProfile], project });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByText("Advanced options"));
    await view.user.click(screen.getByRole("button", { name: "+ cloud" }));
    await view.user.click(screen.getByRole("button", { name: "+ local" }));
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
    expect(
      view.requests.filter((request) => request.method === "PUT"),
    ).toHaveLength(0);
    await view.user.click(
      screen.getByRole("button", { name: "Remove label 2" }),
    );
    await view.user.selectOptions(
      screen.getByLabelText("Time limit"),
      "custom",
    );
    await view.user.clear(screen.getByLabelText("Time limit in hours"));
    await view.user.type(screen.getByLabelText("Time limit in hours"), "9000");
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    await view.user.click(
      screen.getByRole("button", { name: "Save as draft" }),
    );
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
    expect(await view.sent("POST", "/audits")[0]!.json()).toMatchObject({
      labels: { cloud: "" },
    });
    expect(view.sent("POST", "/start")).toHaveLength(0);
  });

  it("preserves shared options when switching between generic and Web scan forms", async () => {
    const view = renderStart(start, {
      profiles: [webScanProfile, top10],
      project,
      materials: [sourceZip],
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByText("Advanced options"));
    await view.user.click(screen.getByRole("button", { name: "+ local" }));
    await view.user.type(screen.getByLabelText("Label value 1"), "lab");
    await view.user.click(screen.getByText("For admins and debugging."));
    await view.user.type(
      screen.getByLabelText("Runtime configuration labels"),
      "debug",
    );
    await act(async () => {
      await view.router.navigate(
        `/checks/new?project=project_shop&type=${top10.ref.name}`,
      );
    });
    expect(await screen.findByLabelText("Label value 1")).toHaveValue("lab");
    expect(screen.getByLabelText("Runtime configuration labels")).toHaveValue(
      "debug",
    );
    await act(async () => {
      await view.router.navigate(start);
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
    expect(await view.sent("POST", "/audits")[0]!.json()).toMatchObject({
      labels: { local: "lab" },
      runtimeLabels: ["debug"],
    });
  });

  it.each(["type", "project", "leave"])(
    "joins pending preparation and does not launch after %s navigation",
    async (destination) => {
      const upload = deferred<Response>();
      const view = renderStart(start, {
        profiles: [webScanProfile, top10],
        project,
        materials: [sourceZip],
        upload: () => upload.promise,
      });
      await screen.findByLabelText(/Target URL/);
      await view.user.click(
        screen.getByRole("button", { name: "Start check" }),
      );
      await waitFor(() =>
        expect(
          view.requests.filter((request) => request.method === "PUT"),
        ).toHaveLength(1),
      );
      expect(screen.getByLabelText("Time limit")).toBeDisabled();
      await view.user.keyboard("{Control>}{Enter}{/Control}");
      expect(
        view.requests.filter((request) => request.method === "PUT"),
      ).toHaveLength(1);
      const href =
        destination === "type"
          ? `/checks/new?project=project_shop&type=${top10.ref.name}`
          : destination === "project"
            ? "/checks/new?project=project_other&type=web-scan"
            : "/projects";
      await act(async () => {
        await view.router.navigate(href);
      });
      const request = view.requests.find(
        (request) => request.method === "PUT",
      )!;
      const name = new URL(request.url).pathname.split("/").at(-1)!;
      const metadata = artifactMetadata("scan-inputs", name, {
        mediaType: "application/json",
        size: (await request.arrayBuffer()).byteLength,
      });
      await act(async () => {
        upload.resolve(
          json(artifactWrite(metadata), 201, {
            ETag: `"${metadata.artifact.revision}"`,
          }),
        );
      });
      expect(
        view.requests.filter((request) => request.method === "PUT"),
      ).toHaveLength(1);
      expect(view.sent("POST", "/audits")).toHaveLength(0);
      expect(
        view.router.state.location.pathname + view.router.state.location.search,
      ).toBe(href);
    },
  );

  it("retries an ambiguous start with its original options after the form changes", async () => {
    let attempts = 0;
    const view = renderStart(start, {
      profiles: [webScanProfile],
      project,
      start: () => {
        if (++attempts === 1) throw new TypeError("Lost response");
        return json(startResponse(draftAudit(webScanProfile)), 200, {
          ETag: '"2"',
        });
      },
    });
    await screen.findByLabelText(/Target URL/);
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    await screen.findByText("The check may already have started.");
    await view.user.click(screen.getByText("Advanced options"));
    await view.user.click(screen.getByRole("button", { name: "+ cloud" }));
    await view.user.clear(screen.getByLabelText(/Target URL/));
    await view.user.type(
      screen.getByLabelText(/Target URL/),
      "https://different.test/",
    );
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    await view.user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() => expect(view.sent("POST", "/start")).toHaveLength(2));
    expect(view.sent("POST", "/audits")).toHaveLength(1);
    const requests = view.sent("POST", "/start");
    expect(requests[1]!.headers.get("Idempotency-Key")).toBe(
      requests[0]!.headers.get("Idempotency-Key"),
    );
    expect(
      view.requests.filter((request) => request.method === "PUT"),
    ).toHaveLength(1);
  });
  it("agrees with the Server on the shared Nuclei selection fixtures", () => {
    expect(selectionCases.length).toBeGreaterThan(0);
    for (const test of selectionCases)
      expect(
        conformsTo("NucleiRuleSelection", test.parameters),
        test.name,
      ).toBe(test.valid);
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
    expect(Object.keys(request.inputs).sort()).toEqual([
      "engagement",
      "target",
    ]);
    expect(request.scope).toEqual({ target: "https://shop.test/" });
    expect(request.parameters).toEqual({
      template_ids: "",
      tags: "",
      severity: "",
    });
    expect(request.inputs.engagement).toEqual(engagementMaterial.artifact);
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
    expect(writes).toHaveLength(2);
    expect(
      writes.every((request) =>
        new URL(request.url).pathname.startsWith(
          "/v1/projects/project_shop/artifacts/",
        ),
      ),
    ).toBe(true);
    expect(await writes[1]!.text()).toBe("id: project-only-rule\n");
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
    ).toHaveLength(1);
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
      await screen.findByText(/Could not save the check materials/),
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
