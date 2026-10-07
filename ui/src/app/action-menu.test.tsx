import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ActionMenu } from "./action-menu";

describe("ActionMenu", () => {
  it("dismisses by Escape with focus return and by an outside pointer", async () => {
    render(
      <>
        <ActionMenu label="Project actions">
          <button>Refresh</button>
        </ActionMenu>
        <button>Outside</button>
      </>,
    );
    const user = userEvent.setup();
    const trigger = screen.getByLabelText("Project actions");
    await user.click(trigger);
    await user.tab();
    expect(screen.getByRole("button", { name: "Refresh" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(trigger.parentElement).not.toHaveAttribute("open");
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "Outside" }));
    expect(trigger.parentElement).not.toHaveAttribute("open");
    expect(screen.getByRole("button", { name: "Outside" })).toHaveFocus();
  });

  it("places the open menu in viewport coordinates and forgets it on close", async () => {
    render(
      <ActionMenu label="Check actions">
        <button>Delete check</button>
      </ActionMenu>,
    );
    const user = userEvent.setup();
    const trigger = screen.getByLabelText("Check actions");
    const panel = screen.getByRole("button", { name: "Delete check" })
      .parentElement as HTMLElement;
    expect(panel).not.toHaveAttribute("data-placed");
    await user.click(trigger);
    expect(panel).toHaveAttribute("data-placed");
    expect(panel.style.top).not.toBe("");
    expect(panel.style.left).not.toBe("");
    await user.keyboard("{Escape}");
    expect(panel).not.toHaveAttribute("data-placed");
    expect(panel.style.top).toBe("");
  });
});
