import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { Facts } from "./facts";
import { Field } from "./field";

describe("Facts", () => {
  it("lists term and value pairs and leaves out facts without a value", () => {
    const { container } = render(
      <Facts
        aria-label="Check facts"
        layout="row"
        items={[
          { term: "Project", value: "crapi" },
          { term: "Started", value: undefined },
          { term: "Finished", value: null },
          { term: "Stopped", value: false },
          { term: "Findings", value: 0 },
        ]}
      />,
    );
    const list = screen.getByLabelText("Check facts");
    expect(list.tagName).toBe("DL");
    expect(list).toHaveAttribute("data-layout", "row");
    expect(
      Array.from(container.querySelectorAll("dt")).map((dt) => dt.textContent),
    ).toEqual(["Project", "Findings"]);
    expect(within(list).getByText("0")).toBeVisible();
  });

  it("marks the rows layout and a fact that takes a row of its own", () => {
    render(
      <Facts
        aria-label="Configuration"
        layout="rows"
        items={[
          { term: "Gateway", value: "local-litellm" },
          {
            term: "Repository",
            value: "https://example.test/repo.git",
            wide: true,
          },
        ]}
      />,
    );
    const list = screen.getByLabelText("Configuration");
    expect(list).toHaveAttribute("data-layout", "rows");
    const [gateway, repository] = Array.from(list.children);
    expect(gateway).not.toHaveAttribute("data-wide");
    expect(repository).toHaveAttribute("data-wide", "");
  });

  it("renders nothing when no fact has a value", () => {
    const { container } = render(
      <Facts items={[{ term: "Started", value: undefined }]} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("keeps technical facts behind a Technical details disclosure", async () => {
    render(
      <Facts
        items={[{ term: "Project", value: "crapi" }]}
        technical={[
          { term: "Check ID", value: "audit-1" },
          { term: "Revision", value: undefined },
        ]}
      />,
    );
    const summary = screen.getByText("Technical details");
    expect(screen.getByText("audit-1")).not.toBeVisible();
    await userEvent.setup().click(summary);
    expect(screen.getByText("audit-1")).toBeVisible();
    expect(screen.queryByText("Revision")).toBeNull();
  });
});

describe("Field", () => {
  function NameField({ error }: { error?: string }) {
    const [value, setValue] = useState("");
    return (
      <Field
        label="Project name"
        hint="Shown in lists and the rail."
        error={error}
        required
      >
        {(control) => (
          <input
            {...control}
            value={value}
            onChange={(event) => setValue(event.currentTarget.value)}
          />
        )}
      </Field>
    );
  }

  it("labels the control and describes it by its hint", async () => {
    render(<NameField />);
    const input = screen.getByRole("textbox", { name: "Project name" });
    expect(input).toBeRequired();
    expect(input).toHaveAttribute("aria-required", "true");
    expect(input).toHaveAccessibleDescription("Shown in lists and the rail.");
    expect(input).not.toHaveAttribute("aria-invalid");
    await userEvent.setup().type(input, "crapi");
    expect(input).toHaveValue("crapi");
  });

  it("marks the control invalid and announces the error", () => {
    render(<NameField error="Use letters, digits and dashes." />);
    const input = screen.getByRole("textbox", { name: "Project name" });
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toHaveAccessibleDescription(
      "Shown in lists and the rail. Use letters, digits and dashes.",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Use letters, digits and dashes.",
    );
  });

  it("says when a field is optional", () => {
    render(
      <Field label="Description" optional id="description">
        {(control) => <textarea {...control} />}
      </Field>,
    );
    const textarea = screen.getByRole("textbox", {
      name: "Description (optional)",
    });
    expect(textarea).toHaveAttribute("id", "description");
    expect(textarea).not.toBeRequired();
  });
});
