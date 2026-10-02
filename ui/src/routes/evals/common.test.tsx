import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { KeyValueEditor } from "./common";
import {
  KeyValueValidityProvider,
  useKeyValueValidity,
} from "./key-value-validity";

function EditorHarness({
  initial,
  save,
}: {
  initial: Record<string, string>;
  save: (value: Record<string, string>) => void;
}) {
  const [value, setValue] = useState(initial);
  const validity = useKeyValueValidity();
  return (
    <KeyValueValidityProvider value={validity.register}>
      <KeyValueEditor label="Parameters" value={value} onChange={setValue} />
      <button
        type="button"
        disabled={validity.invalid}
        onClick={() => save(value)}
      >
        Save
      </button>
    </KeyValueValidityProvider>
  );
}

describe("Eval key/value editor", () => {
  it("keeps a duplicate draft visible until the full prefix key is typed", async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    render(
      <EditorHarness initial={{ target: "https://a", "": "" }} save={save} />,
    );
    const name = screen.getByRole("textbox", { name: "Parameters name 2" });
    const button = screen.getByRole("button", { name: "Save" });

    expect(button).toBeDisabled();
    await user.type(name, "target");
    expect(name).toHaveValue("target");
    expect(name).toHaveAttribute("aria-invalid", "true");
    expect(screen.getAllByRole("alert")[0]).toHaveTextContent(
      "This name is already in use",
    );
    expect(button).toBeDisabled();
    await user.type(name, "_url");
    expect(name).toHaveValue("target_url");
    expect(name).toHaveAttribute("aria-invalid", "false");
    expect(button).toBeEnabled();
    await user.click(button);
    expect(save).toHaveBeenCalledWith({ target: "https://a", target_url: "" });
  });

  it("adds a free field name without overwriting an existing entry", async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    render(
      <EditorHarness initial={{ "field-3": "keep-me", "": "" }} save={save} />,
    );
    await user.click(screen.getByRole("button", { name: "Add parameters" }));
    expect(
      screen.getByRole("textbox", { name: "Parameters name 3" }),
    ).toHaveValue("field-1");
    expect(
      screen.getByRole("textbox", { name: "Parameters value 1" }),
    ).toHaveValue("keep-me");
    await user.type(
      screen.getByRole("textbox", { name: "Parameters name 2" }),
      "other",
    );
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(save).toHaveBeenCalledWith({
      "field-3": "keep-me",
      other: "",
      "field-1": "",
    });
    await user.click(screen.getByRole("button", { name: "Add parameters" }));
    expect(
      screen.getByRole("textbox", { name: "Parameters name 4" }),
    ).toHaveValue("field-2");
  });

  it("keeps a numeric key in its row and retains focus", async () => {
    const user = userEvent.setup();
    const save = vi.fn();
    render(<EditorHarness initial={{ a: "x", "": "y" }} save={save} />);
    const numericName = screen.getByRole("textbox", {
      name: "Parameters name 2",
    });
    await user.type(numericName, "12");
    expect(
      screen.getByRole("textbox", { name: "Parameters name 1" }),
    ).toHaveValue("a");
    expect(numericName).toHaveValue("12");
    expect(numericName).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(save).toHaveBeenCalledWith({ a: "x", "12": "y" });
  });
});
