import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderApplication } from "../../../test/application";
import {
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
  researchFixture,
} from "../../../test/audit-research-fixtures";
import { fakeServer } from "../../../test/fake-server";
import { json } from "../../../test/responses";
import { scopedTaskResult } from "./task-result";

const root = `/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}`;
function render(section = "", state = "normal") {
  const fixture = researchFixture(state);
  const server = fakeServer({}, (request) => fixture.respond(request));
  renderApplication(root + section, { server });
  return fixture;
}

describe("combined Checks research workspace", () => {
  it("distinguishes a system interruption from a model judgment", async () => {
    render("/work?task=task_7", "interrupted");
    const record = await screen.findByRole("region", {
      name: "System interruption",
    });
    expect(
      within(record).getByRole("heading", { name: "No accepted conclusion" }),
    ).toBeInTheDocument();
    expect(within(record).getByText("System record")).toBeInTheDocument();
    expect(
      within(record).getByText("The configured model call limit was reached"),
    ).toBeInTheDocument();
    expect(within(record).getByText("Execution record")).toBeInTheDocument();
    expect(within(record).queryByText("Inconclusive")).not.toBeInTheDocument();
    expect(
      within(record).queryByText("Conditions reported by the model"),
    ).not.toBeInTheDocument();
  });
  it("reads the exact previous attempt through its retained key after source deletion", async () => {
    render("/work?task=task_0", "interrupted");
    await userEvent.click(
      await screen.findByRole("button", { name: "Read saved result" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Recorded result" }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByText("Run attempts"));
    const pages = await screen.findByRole("navigation", {
      name: "Attempt pages",
    });
    await userEvent.click(within(pages).getByRole("button", { name: "Next" }));
    const old = await screen.findByRole("listitem", { name: "Attempt 1" });
    expect(old).toHaveTextContent("failed");
    expect(old).toHaveTextContent("source Run deleted");
    await userEvent.click(
      within(old).getByRole("button", { name: "Read saved result" }),
    );
    expect(
      await within(old).findByText(
        "Earlier accepted assessment from a separate attempt",
      ),
    ).toBeInTheDocument();
    expect(
      within(old).getByText("The old target conditions may have changed"),
    ).toBeInTheDocument();
  });
  it("does not infer execution or a system judgment for work with no recorded activity", async () => {
    render("/work?task=task_6", "interrupted");
    expect(
      await screen.findByText("No recorded activity for this work item."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "System interruption" }),
    ).not.toBeInTheDocument();
  });
  it("offers Audit-local SARIF export beside canonical Project findings", async () => {
    render("/findings");
    const summary = await screen.findByText("Export SARIF", {
      selector: "summary",
    });
    await userEvent.click(summary);
    expect(
      screen.getByRole("button", { name: "Create SARIF export" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("checkbox", { name: "Include duplicates" }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("link", { name: "All Project issues →" }),
    ).toHaveAttribute("href", `/projects/${RESEARCH_PROJECT}/findings`);
  });
  it("moves selection and focus together with the existing list keyboard controls", async () => {
    render("/work");
    await userEvent.click(
      await screen.findByRole("button", {
        name: /A customer can read another customer/,
      }),
    );
    await userEvent.keyboard("j");
    expect(
      screen.getByRole("button", { name: /Product search input can change/ }),
    ).toHaveFocus();
    expect(
      screen.getByRole("heading", {
        name: "Product search input can change the SQL query",
      }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Home}");
    expect(
      screen.getByRole("button", {
        name: /A customer can read another customer/,
      }),
    ).toHaveFocus();
  });
  it("keeps an exact work input selected while browsing a second material page", async () => {
    const fixture = researchFixture();
    const original = fixture.materials[0]!;
    const entries = Array.from({ length: 51 }, (_, index) => ({
      ...structuredClone(original),
      key: `page-material-${index}`,
      material: {
        ...original.material,
        ref: {
          ...original.material.ref,
          name: `material-${index}`,
          revision: `material-${index}-r1`,
        },
      },
    }));
    const server = fakeServer({}, async (request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/materials"))
        return json({
          items: url.searchParams.has("cursor")
            ? entries.slice(50)
            : entries.slice(0, 50),
          page: url.searchParams.has("cursor")
            ? { hasMore: false }
            : { hasMore: true, nextCursor: "materials-50" },
          auditRevision: 3,
        });
      return fixture.respond(request);
    });
    renderApplication(root + "/work", { server });
    await userEvent.click(
      await screen.findByRole("button", { name: "Add work" }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.type(
      within(dialog).getByRole("textbox", { name: /What should/ }),
      "Inspect selected material",
    );
    await userEvent.selectOptions(
      await within(dialog).findByRole("combobox", { name: /Method/ }),
      "inspect",
    );
    await userEvent.selectOptions(
      await within(dialog).findByRole("combobox", {
        name: /Material for source/,
      }),
      "page-material-0",
    );
    await userEvent.click(within(dialog).getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(
        within(dialog).getByRole("option", { name: /material-50@/ }),
      ).toBeInTheDocument(),
    );
    expect(
      within(dialog).getByRole("combobox", { name: /Material for source/ }),
    ).toHaveValue("page-material-0");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Accept work" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    const write = fixture.calls.find(
      (c) => c.method === "POST" && c.path.endsWith("/tasks"),
    );
    expect(write?.body).toMatchObject({
      work: [{ inputs: { source: entries[0]!.material } }],
    });
  });
  it("uses whole-check progress without hydrating task outputs or claiming semantic coverage", async () => {
    const fixture = render();
    expect(await screen.findByText("8 accepted tasks")).toBeInTheDocument();
    expect(
      await screen.findByText("4 original requirements"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("navigation", { name: "Check sections" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Report →" })).toHaveAttribute(
      "href",
      `/reports/${RESEARCH_AUDIT}`,
    );
    expect(
      fixture.calls.filter((c) => c.path.endsWith("/artifacts")),
    ).toHaveLength(0);
    expect(
      fixture.calls.filter(
        (c) => c.path.endsWith("/coverage") || c.path.endsWith("/reviews"),
      ),
    ).toHaveLength(0);
  });
  it("reads only the selected exact task member and shows its scoped result", async () => {
    const fixture = render("/work");
    await userEvent.click(
      await screen.findByRole("button", { name: "Read saved result" }),
    );
    expect(
      await screen.findByText("Supported", { exact: true }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Only the selected endpoint and baseline were tested"),
    ).toBeInTheDocument();
    expect(
      fixture.calls.filter((c) => c.path.endsWith("/artifacts")),
    ).toHaveLength(1);
    expect(
      screen.queryByText("Refuted", { exact: true }),
    ).not.toBeInTheDocument();
  });
  it("keeps refuted and inconclusive checks in hypothesis history, apart from Project decisions", async () => {
    render("/hypotheses");
    await userEvent.click(
      await screen.findByRole("button", {
        name: /Product search input can change the SQL query/,
      }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Read saved result" }),
    );
    expect(
      await screen.findByText("Refuted", { exact: true }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Confirm issue" }),
    ).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", {
        name: /A refresh token can be reused across sessions/,
      }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Read saved result" }),
    );
    expect(
      await screen.findByText("Inconclusive", { exact: true }),
    ).toBeInTheDocument();
  });
  it("opens exact grounds through accepted material receipts and never substitutes latest", async () => {
    const fixture = render("/hypotheses");
    const source = await screen.findByRole("link", {
      name: /order-api-observations/,
    });
    expect(source.getAttribute("href")).toContain(
      encodeURIComponent("order-api-observations-r1"),
    );
    await userEvent.click(source);
    await userEvent.click(
      (await screen.findAllByRole("button", { name: "Preview" }))[0]!,
    );
    expect(
      await screen.findByText(/Order lookup and invoice exports/),
    ).toBeInTheDocument();
    const reads = fixture.calls.filter((c) => c.path.endsWith("/artifacts"));
    expect(reads).toHaveLength(1);
  });
  it("reconciles an uncertain exact publication with the original body and key", async () => {
    const fixture = render("/materials", "ambiguous");
    await userEvent.click(
      (
        await screen.findAllByRole("button", {
          name: "Publish to Project",
        })
      )[0]!,
    );
    const dialog = screen.getByRole("dialog");
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Publish exact revision" }),
    );
    await waitFor(() =>
      expect(
        within(dialog).getByRole("textbox", { name: /Material name/ }),
      ).toBeDisabled(),
    );
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Retry publication" }),
    );
    expect(await screen.findByText("Material published.")).toBeInTheDocument();
    const writes = fixture.calls.filter((c) =>
      c.path.endsWith("/audit-material-publications"),
    );
    expect(writes).toHaveLength(2);
    expect(writes[0]?.key).toBe(writes[1]?.key);
    expect(writes[0]?.body).toEqual(writes[1]?.body);
  });
  it("retries accepted ordinary work without changing its identity or exact inputs", async () => {
    const fixture = render("/work", "ambiguous");
    await userEvent.click(
      await screen.findByRole("button", { name: "Add work" }),
    );
    await userEvent.type(
      await screen.findByRole("textbox", {
        name: /What should this work investigate/,
      }),
      "Explore invoice access",
    );
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: /Method/ }),
      "inspect",
    );
    await userEvent.selectOptions(
      await screen.findByRole("combobox", { name: /Material for source/ }),
      "material/order-api-observations",
    );
    await userEvent.click(screen.getByRole("button", { name: "Accept work" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Retry accepting work" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    const writes = fixture.calls.filter(
      (c) => c.path.endsWith("/tasks") && c.method === "POST",
    );
    expect(writes).toHaveLength(2);
    expect(writes[0]?.key).toBe(writes[1]?.key);
    expect(writes[0]?.body).toEqual(writes[1]?.body);
  });
});

it("rejects a plausible result that belongs to another exact assertion revision", () => {
  const fixture = researchFixture();
  const task = fixture.tasks[0]!;
  const text = fixture.payloads.get("task/task_0/result")!;
  expect(scopedTaskResult(text, task)?.verdict).toBe("supported");
  const changed = structuredClone(task);
  changed.document.hypothesis!.revision++;
  expect(() => scopedTaskResult(text, changed)).toThrow(
    "exact hypothesis context",
  );
});
