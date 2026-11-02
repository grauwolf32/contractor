import { webcrypto } from "node:crypto";
import { act, screen, waitFor } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import fixtures from "../../../../../api/testdata/audit-checklist-authoring.json";
import limits from "../../../../../api/testdata/audit-common/boundaries.json";
import {
  artifactMetadata,
  artifactWrite,
  conforming,
} from "../../../test/builders";
import { deferred } from "../../../test/deferred";
import { renderWithAPI } from "../../../test/render";
import { json } from "../../../test/responses";
import {
  authoredChecklistEntry,
  checklistDocument,
  checklistStatementError,
  CHECKLIST_DOCUMENT_BYTES,
  CHECKLIST_TEXT_BYTES,
} from "./checklist-document";
import { useChecklistDraft, type ChecklistDraft } from "./checklist-state";
import { checklist, sourceZip, top10 } from "./test-fixtures";
import { renderStart } from "./test-support";

beforeAll(() => vi.stubGlobal("crypto", webcrypto));
afterAll(() => vi.unstubAllGlobals());
const profile = conforming("AuditProfile", {
  ...checklist,
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
    checklist: { required: true, mediaTypes: ["application/json"] },
  },
});
const start = `/checks/new?project=project_shop&type=${profile.ref.name}`;
async function openEditor(view: ReturnType<typeof renderStart>) {
  await view.user.click(
    await screen.findByRole("button", { name: "Write checklist" }),
  );
  return screen.getByRole("textbox", { name: "Item 1" });
}
async function apply(view: ReturnType<typeof renderStart>) {
  const field = await openEditor(view);
  await view.user.type(field, "Check tenant isolation.");
  await view.user.click(screen.getByRole("button", { name: "Use checklist" }));
}
const puts = (view: ReturnType<typeof renderStart>) =>
  view.requests.filter((request) => request.method === "PUT");

describe("authored checklist contract", () => {
  it.each(fixtures)("matches shared domain fixture: $name", ({ document }) => {
    expect(
      checklistDocument(
        document.items.map((item) =>
          authoredChecklistEntry(item.key, item.statement, item.version),
        ),
      ),
    ).toEqual(document);
  });
  it("uses shared UTF-8/document bounds and rejects invalid/empty text", () => {
    expect(CHECKLIST_TEXT_BYTES).toBe(limits.text_bytes);
    expect(CHECKLIST_DOCUMENT_BYTES).toBe(limits.document_bytes);
    expect(
      checklistStatementError("界".repeat(Math.floor(limits.text_bytes / 3))),
    ).toBeUndefined();
    expect(
      checklistStatementError(
        "界".repeat(Math.floor(limits.text_bytes / 3) + 1),
      ),
    ).toMatch(/64 KiB/);
    for (const text of ["  \n", "bad\0text", "\ud800"])
      expect(checklistStatementError(text)).toBeDefined();
  });
  it("keeps stable keys/versions, resets restored edits, and bounds the composer at 128", async () => {
    let draft!: ChecklistDraft;
    function Harness() {
      draft = useChecklistDraft("project_shop", {
        ...profile,
        execution: { ...profile.execution, maxItemsTotal: 200 },
      });
      return null;
    }
    renderWithAPI(<Harness />);
    expect(draft.entries).toHaveLength(1);
    expect(draft.entries[0]?.statement).toBe("");
    act(() => draft.setMode("write"));
    act(() => draft.update("check-0001", "First check"));
    act(() => draft.apply());
    act(() => draft.update("check-0001", "Changed"));
    expect(draft.entries[0]?.version).toBe("2");
    expect(draft.blocker).toMatch(/Apply/);
    act(() => draft.update("check-0001", "First check"));
    expect(draft.dirty).toBe(false);
    act(() => draft.add());
    act(() => draft.remove("check-0001"));
    expect(draft.entries[0]?.key).toBe("check-0002");
    for (let i = 0; i < 130; i++) act(() => draft.add());
    expect(draft.entries).toHaveLength(128);
    expect(new Set(draft.entries.map((entry) => entry.key)).size).toBe(128);
  });
  it("uses the lower profile bound and never applies an invalid document", () => {
    let draft!: ChecklistDraft;
    function Harness() {
      draft = useChecklistDraft("project_shop", {
        ...profile,
        execution: { ...profile.execution, maxItemsTotal: 1 },
      });
      return null;
    }
    renderWithAPI(<Harness />);
    act(() => draft.setMode("write"));
    act(() => draft.add());
    act(() => draft.apply());
    expect(draft.entries).toHaveLength(1);
    expect(draft.applied).toBeUndefined();
  });
});

describe("real New check authoring path", () => {
  it.each(["Start check", "Save as draft"])(
    "Apply stays local; %s retains an ordinary exact material before creating",
    async (intent) => {
      const view = renderStart(start, {
        profiles: [profile],
        materials: [sourceZip],
      });
      await apply(view);
      expect(puts(view)).toHaveLength(0);
      expect(view.sent("POST", "/audits")).toHaveLength(0);
      expect(
        screen.getByText(/Will be saved as a project material/),
      ).toBeVisible();
      await view.user.click(screen.getByRole("button", { name: intent }));
      await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
      expect(puts(view)).toHaveLength(1);
      const put = puts(view)[0]!;
      expect(put.headers.get("If-None-Match")).toBe("*");
      expect(put.headers.get("If-Match")).toBeNull();
      const doc = await put.clone().json();
      expect(doc).toEqual(
        checklistDocument([
          authoredChecklistEntry("check-0001", "Check tenant isolation."),
        ]),
      );
      const body = await view.sent("POST", "/audits")[0]!.clone().json();
      expect(body.inputs.source).toEqual(sourceZip.artifact);
      expect(body.inputs.checklist).toMatchObject({
        namespace: "check-inputs",
        name: expect.stringMatching(/^checklist\.[0-9a-f]{64}$/),
        revision: expect.any(String),
      });
      expect(
        new URL(put.url).pathname.endsWith(body.inputs.checklist.name),
      ).toBe(true);
      await waitFor(() =>
        expect(view.router.state.location.pathname).toContain(
          "/audits/audit_new",
        ),
      );
      expect(view.sent("POST", "/start")).toHaveLength(
        intent === "Start check" ? 1 : 0,
      );
    },
  );
  it("blocks dirty edits and restores the applied version without writing", async () => {
    const view = renderStart(start, {
      profiles: [profile],
      materials: [sourceZip],
    });
    await apply(view);
    await view.user.type(screen.getByLabelText("Item 1"), " More.");
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
    await view.user.click(
      screen.getByRole("button", { name: "Use checklist" }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Save as draft" }),
    );
    await waitFor(() => expect(puts(view)).toHaveLength(1));
    expect((await puts(view)[0]!.clone().json()).items[0]).toMatchObject({
      key: "check-0001",
      version: "2",
    });
  });
  it.each([false, true])(
    "reconciles a lost retention answer only for exact bytes (mismatch=%s)",
    async (mismatch) => {
      let saved!: ReturnType<typeof artifactMetadata>;
      let content = "";
      const view = renderStart(start, {
        profiles: [profile],
        materials: [sourceZip],
        upload: async (request, url) => {
          content = await request.text();
          saved = artifactMetadata(
            "check-inputs",
            url.pathname.split("/").at(-1)!,
            {
              mediaType: "application/json",
              size: new TextEncoder().encode(content).byteLength,
            },
          );
          throw new TypeError("lost answer");
        },
        metadata: () =>
          json(saved, 200, { ETag: `"${saved.artifact.revision}"` }),
        content: () =>
          new Response(
            mismatch ? content.replace("tenant", "Tenant") : content,
            {
              headers: {
                "content-type": "application/json",
                "X-Contractor-API-Version": "contractor.public.v1",
              },
            },
          ),
      });
      await apply(view);
      await view.user.click(
        screen.getByRole("button", { name: "Start check" }),
      );
      if (mismatch) {
        expect(
          await screen.findByText(/Could not save the check materials/),
        ).toBeVisible();
        expect(view.sent("POST", "/audits")).toHaveLength(0);
        expect(screen.getByLabelText("Item 1")).toHaveValue(
          "Check tenant isolation.",
        );
      } else
        await waitFor(() =>
          expect(view.sent("POST", "/start")).toHaveLength(1),
        );
      expect(puts(view)).toHaveLength(1);
    },
  );
  it("abandons pending retention on type navigation without creating a check", async () => {
    const pending = deferred<Response>();
    let name = "",
      size = 0;
    const view = renderStart(start, {
      profiles: [profile, top10],
      materials: [sourceZip],
      upload: async (request, url) => {
        name = url.pathname.split("/").at(-1)!;
        size = (await request.arrayBuffer()).byteLength;
        return pending.promise;
      },
    });
    await apply(view);
    await view.user.click(screen.getByRole("button", { name: "Start check" }));
    await waitFor(() => expect(puts(view)).toHaveLength(1));
    await act(() =>
      view.router.navigate(
        `/checks/new?project=project_shop&type=${top10.ref.name}`,
      ),
    );
    const metadata = artifactMetadata("check-inputs", name, {
      mediaType: "application/json",
      size,
    });
    await act(async () => {
      pending.resolve(
        json(artifactWrite(metadata), 201, {
          ETag: `"${metadata.artifact.revision}"`,
        }),
      );
      await pending.promise;
    });
    expect(view.sent("POST", "/audits")).toHaveLength(0);
    expect(view.sent("POST", "/start")).toHaveLength(0);
  });
});
