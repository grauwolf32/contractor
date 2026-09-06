import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useId, useRef, useState, type FormEvent, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { Dialog } from "./dialog";

function BasicDialogHarness() {
  const [open, setOpen] = useState(false);
  const heading = useId();
  const initialFocus = useRef<HTMLInputElement>(null);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open editor
      </button>
      <button type="button">Background action</button>
      {open ? (
        <Dialog
          className="project-dialog"
          labelledBy={heading}
          initialFocusRef={initialFocus}
          onRequestClose={() => setOpen(false)}
        >
          <h2 id={heading}>Editor</h2>
          <input ref={initialFocus} aria-label="First field" />
          <button type="button">Last action</button>
        </Dialog>
      ) : null}
    </>
  );
}

function ParentDialog({ children }: { children: ReactNode }) {
  const heading = useId();
  return (
    <Dialog
      className="project-dialog"
      labelledBy={heading}
      onRequestClose={vi.fn()}
    >
      <h2 id={heading}>Parent</h2>
      {children}
    </Dialog>
  );
}

describe("Dialog", () => {
  it("makes the background inert, contains focus and restores its trigger", async () => {
    const view = render(<BasicDialogHarness />);
    const user = userEvent.setup();
    const trigger = screen.getByRole("button", { name: "Open editor" });
    const background = screen.getByRole("button", {
      name: "Background action",
    });

    await user.click(trigger);
    const dialog = await screen.findByRole("dialog", { name: "Editor" });
    const first = screen.getByRole("textbox", { name: "First field" });
    const last = screen.getByRole("button", { name: "Last action" });
    await waitFor(() => expect(first).toHaveFocus());
    expect(view.container).toHaveAttribute("inert");
    expect(view.container).toHaveAttribute("aria-hidden", "true");
    expect(document.body.style.overflow).toBe("hidden");

    background.focus();
    expect(first).toHaveFocus();
    last.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(first).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(last).toHaveFocus();

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
    expect(view.container).not.toHaveAttribute("inert");
    expect(view.container).not.toHaveAttribute("aria-hidden");
    expect(document.body.style.overflow).toBe("");
  });

  it("keeps only the newest nested layer interactive and isolates child form submission", async () => {
    const parentSubmit = vi.fn();
    const childSubmit = vi.fn();

    function Harness() {
      const [parentOpen, setParentOpen] = useState(false);
      const [childOpen, setChildOpen] = useState(false);
      const parentHeading = useId();
      const childHeading = useId();
      return (
        <>
          <button type="button" onClick={() => setParentOpen(true)}>
            Start setup
          </button>
          {parentOpen ? (
            <Dialog
              className="project-dialog"
              labelledBy={parentHeading}
              onRequestClose={() => setParentOpen(false)}
            >
              <h2 id={parentHeading}>Run setup</h2>
              <form
                aria-label="Parent form"
                onSubmit={(event) => {
                  event.preventDefault();
                  parentSubmit();
                }}
              >
                <button type="button" onClick={() => setChildOpen(true)}>
                  Import source
                </button>
                {childOpen ? (
                  <Dialog
                    className="project-dialog"
                    labelledBy={childHeading}
                    onRequestClose={() => setChildOpen(false)}
                  >
                    <h2 id={childHeading}>Import source</h2>
                    <form
                      aria-label="Child form"
                      onSubmit={(event: FormEvent) => {
                        event.preventDefault();
                        childSubmit();
                      }}
                    >
                      <input aria-label="Repository" />
                      <button type="submit">Import</button>
                    </form>
                  </Dialog>
                ) : null}
              </form>
            </Dialog>
          ) : null}
        </>
      );
    }

    const user = userEvent.setup();
    render(<Harness />);
    const start = screen.getByRole("button", { name: "Start setup" });
    await user.click(start);
    const parent = await screen.findByRole("dialog", { name: "Run setup" });
    const childTrigger = screen.getByRole("button", { name: "Import source" });
    await user.click(childTrigger);
    const child = await screen.findByRole("dialog", { name: "Import source" });
    const parentLayer = parent.closest("[data-contractor-dialog-layer]");
    expect(parentLayer).toHaveAttribute("inert");
    expect(parentLayer).toHaveAttribute("aria-hidden", "true");

    await user.click(screen.getByRole("button", { name: "Import" }));
    expect(childSubmit).toHaveBeenCalledOnce();
    expect(parentSubmit).not.toHaveBeenCalled();

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(child).not.toBeInTheDocument());
    expect(parent).toBeInTheDocument();
    await waitFor(() => expect(childTrigger).toHaveFocus());
    expect(parentLayer).not.toHaveAttribute("inert");

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(parent).not.toBeInTheDocument());
    await waitFor(() => expect(start).toHaveFocus());
  });

  it("requests close without owning it and falls back when a nested trigger disappears", async () => {
    const requests = vi.fn();

    function Harness() {
      const [childOpen, setChildOpen] = useState(false);
      const [showTrigger, setShowTrigger] = useState(true);
      const [allowClose, setAllowClose] = useState(false);
      const heading = useId();
      return (
        <ParentDialog>
          {showTrigger ? (
            <button type="button" onClick={() => setChildOpen(true)}>
              Child trigger
            </button>
          ) : null}
          <button type="button">Parent fallback</button>
          {childOpen ? (
            <Dialog
              className="project-dialog"
              labelledBy={heading}
              onRequestClose={() => {
                requests();
                if (allowClose) {
                  setShowTrigger(false);
                  setChildOpen(false);
                }
              }}
            >
              <h2 id={heading}>Guarded child</h2>
              <button type="button" onClick={() => setAllowClose(true)}>
                Allow close
              </button>
            </Dialog>
          ) : null}
        </ParentDialog>
      );
    }

    render(<Harness />);
    await userEvent.click(
      screen.getByRole("button", { name: "Child trigger" }),
    );
    const child = await screen.findByRole("dialog", { name: "Guarded child" });
    fireEvent.keyDown(document, { key: "Escape" });
    expect(requests).toHaveBeenCalledOnce();
    expect(child).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Allow close" }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(requests).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(child).not.toBeInTheDocument());
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Parent fallback" }),
      ).toHaveFocus(),
    );
  });
});
