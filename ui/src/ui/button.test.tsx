import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { Icon } from "../app/icon";
import { Button, ButtonLink, IconButton } from "./button";

describe("Button", () => {
  it("is a plain button that never submits by accident", async () => {
    const onClick = vi.fn();
    render(
      <form onSubmit={() => onClick("submitted")}>
        <Button onClick={() => onClick("clicked")}>Save draft</Button>
      </form>,
    );
    const button = screen.getByRole("button", { name: "Save draft" });
    expect(button).toHaveAttribute("type", "button");
    expect(button).toHaveClass("ui-btn");
    expect(button).not.toHaveAttribute("data-variant");
    expect(button).not.toHaveAttribute("data-size");
    await userEvent.setup().click(button);
    expect(onClick).toHaveBeenCalledExactlyOnceWith("clicked");
  });

  it("maps variant and size onto the ui-btn data attributes", () => {
    render(
      <>
        <Button variant="primary" size="sm">
          Start check
        </Button>
        <Button variant="danger" size="xs" type="submit">
          Delete
        </Button>
        <Button variant="link">Show all</Button>
      </>,
    );
    const start = screen.getByRole("button", { name: "Start check" });
    expect(start).toHaveAttribute("data-variant", "primary");
    expect(start).toHaveAttribute("data-size", "sm");
    const remove = screen.getByRole("button", { name: "Delete" });
    expect(remove).toHaveAttribute("type", "submit");
    expect(remove).toHaveAttribute("data-variant", "danger");
    expect(screen.getByRole("button", { name: "Show all" })).toHaveAttribute(
      "data-variant",
      "link",
    );
  });

  it("is disabled and busy while pending, with its pending label", () => {
    const { rerender } = render(
      <Button pending pendingLabel="Saving…" icon={<Icon name="plus" />}>
        Save
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Saving…" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    rerender(<Button pending={false}>Save</Button>);
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save" })).not.toHaveAttribute(
      "aria-busy",
    );
  });

  it("passes a ref to the button element", () => {
    const ref = { current: null as HTMLButtonElement | null };
    render(<Button ref={ref}>Focus me</Button>);
    expect(ref.current).toBe(screen.getByRole("button", { name: "Focus me" }));
  });
});

describe("IconButton", () => {
  it("is named and titled by its label", () => {
    render(
      <IconButton
        label="Delete Run run-1"
        icon={<Icon name="delete" />}
        variant="ghost"
        size="sm"
      />,
    );
    const button = screen.getByRole("button", { name: "Delete Run run-1" });
    expect(button).toHaveAttribute("title", "Delete Run run-1");
    expect(button).toHaveAttribute("data-icon-only", "");
    expect(button).toHaveAttribute("data-variant", "ghost");
    expect(button).toHaveTextContent("");
  });
});

describe("ButtonLink", () => {
  function renderAt(element: React.ReactNode) {
    const router = createMemoryRouter(
      [
        { path: "/", element },
        { path: "/checks/new", element: <p>Start page</p> },
      ],
      { initialEntries: ["/"] },
    );
    render(<RouterProvider router={router} />);
    return router;
  }

  it("is a link styled as a button", async () => {
    const router = renderAt(
      <ButtonLink to="/checks/new" variant="primary" size="sm">
        Start a check
      </ButtonLink>,
    );
    const link = screen.getByRole("link", { name: "Start a check" });
    expect(link).toHaveClass("ui-btn");
    expect(link).toHaveAttribute("data-variant", "primary");
    await userEvent.setup().click(link);
    expect(router.state.location.pathname).toBe("/checks/new");
  });

  it("offers the way back with returnLabel", async () => {
    const router = renderAt(
      <ButtonLink to="/checks/new" returnLabel="Inbox">
        Start a check
      </ButtonLink>,
    );
    await userEvent
      .setup()
      .click(screen.getByRole("link", { name: "Start a check" }));
    expect(router.state.location.state).toMatchObject({
      returnLabel: "Inbox",
      returnTo: "/",
    });
  });
});
