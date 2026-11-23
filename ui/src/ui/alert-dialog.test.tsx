import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { AlertDialog } from "./alert-dialog";

function Harness({
  pending = false,
  onConfirm = () => undefined,
  closeLabel,
  dismissOnBackdrop,
}: {
  pending?: boolean;
  onConfirm?: () => void;
  closeLabel?: string;
  dismissOnBackdrop?: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Delete check
      </button>
      {open ? (
        <AlertDialog
          eyebrow="Permanent action"
          title="Delete this check?"
          confirmLabel="Delete check"
          pendingLabel="Deleting…"
          pending={pending}
          closeLabel={closeLabel}
          dismissOnBackdrop={dismissOnBackdrop}
          details={<p>Check audit-1</p>}
          error={pending ? null : <p role="alert">Could not delete.</p>}
          onConfirm={onConfirm}
          onCancel={() => setOpen(false)}
        >
          <p>This cannot be undone.</p>
        </AlertDialog>
      ) : null}
    </>
  );
}

describe("AlertDialog", () => {
  it("is a named, described alert dialog that starts on the safe action", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete this check?",
    });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleDescription("This cannot be undone.");
    expect(dialog).toHaveClass("ui-alert-dialog-danger");
    expect(within(dialog).getByText("Check audit-1")).toBeVisible();
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "Could not delete.",
    );
    expect(
      within(dialog).getByRole("button", { name: "Cancel" }),
    ).toHaveFocus();
  });

  it("returns focus to the opener after Cancel and Escape", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const opener = screen.getByRole("button", { name: "Delete check" });
    await user.click(opener);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await waitFor(() => expect(opener).toHaveFocus());
    await user.click(opener);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("confirms only through its confirm button", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(<Harness onConfirm={onConfirm} />);
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    const dialog = screen.getByRole("alertdialog");
    await user.click(dialog.parentElement!);
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
    await user.click(
      within(dialog).getByRole("button", { name: "Delete check" }),
    );
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("closes from the backdrop and the close button only when asked to", async () => {
    const user = userEvent.setup();
    render(<Harness dismissOnBackdrop closeLabel="Close confirmation" />);
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    await user.click(screen.getByRole("alertdialog").parentElement!);
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    await user.click(
      screen.getByRole("button", { name: "Close confirmation" }),
    );
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("keeps a pending request: every control is disabled and Escape does nothing", async () => {
    const user = userEvent.setup();
    render(<Harness pending closeLabel="Close confirmation" />);
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    const dialog = screen.getByRole("alertdialog");
    expect(
      within(dialog).getByRole("button", { name: "Cancel" }),
    ).toBeDisabled();
    expect(
      within(dialog).getByRole("button", { name: "Close confirmation" }),
    ).toBeDisabled();
    const confirm = within(dialog).getByRole("button", { name: "Deleting…" });
    expect(confirm).toBeDisabled();
    expect(confirm).toHaveAttribute("aria-busy", "true");
    await user.keyboard("{Escape}");
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
  });

  it("uses the primary look for a confirmation that destroys nothing", async () => {
    const user = userEvent.setup();
    function Continue() {
      const [open, setOpen] = useState(true);
      return open ? (
        <AlertDialog
          title="Continue from failed stage?"
          tone="primary"
          cancelLabel="Back"
          confirmLabel="Confirm continuation"
          onConfirm={() => undefined}
          onCancel={() => setOpen(false)}
        >
          Model and tool calls run again.
        </AlertDialog>
      ) : null;
    }
    render(<Continue />);
    const dialog = screen.getByRole("alertdialog", {
      name: "Continue from failed stage?",
    });
    expect(dialog).not.toHaveClass("ui-alert-dialog-danger");
    expect(
      within(dialog).getByRole("button", { name: "Confirm continuation" }),
    ).toHaveAttribute("data-variant", "primary");
    await user.click(within(dialog).getByRole("button", { name: "Back" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});
