import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { StudioRoute } from ".";
import { saveBlob } from "../../app/download";
vi.mock("../../app/download", () => ({ saveBlob: vi.fn() }));

function mount() {
  const router = createMemoryRouter(
    [
      { path: "/catalog/studio", element: <StudioRoute /> },
      { path: "/catalog/workflows", element: <h1>Workflow library</h1> },
    ],
    { initialEntries: ["/catalog/studio"] },
  );
  render(<RouterProvider router={router} />);
  return router;
}
describe("studio draft lifecycle", () => {
  it("guards route changes, preserves a cancelled draft, and exports without browser storage", async () => {
    const user = userEvent.setup();
    mount();
    const storage = vi.spyOn(Storage.prototype, "setItem");
    const objective = screen.getByLabelText("Objective");
    fireEvent.change(objective, { target: { value: "Keep my changes" } });
    fireEvent.blur(objective);
    const unload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(true);
    await user.click(screen.getByRole("link", { name: "Library" }));
    expect(
      screen.getByRole("dialog", { name: "Discard current draft changes?" }),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(screen.getByLabelText("Objective")).toHaveValue("Keep my changes");
    await user.click(screen.getByRole("button", { name: "Export YAML" }));
    expect(saveBlob).toHaveBeenCalledWith(expect.any(Blob), "untitled.yaml");
    expect(storage).not.toHaveBeenCalled();
    storage.mockRestore();
    await user.click(screen.getByRole("link", { name: "Library" }));
    await screen.findByRole("heading", { name: "Workflow library" });
  });
  it("keeps unapplied invalid YAML recoverable and blocks replacement until a concrete discard", async () => {
    const user = userEvent.setup();
    mount();
    await user.click(screen.getByRole("button", { name: "YAML" }));
    fireEvent.change(screen.getByLabelText("Authored YAML"), {
      target: { value: "kind: Workflow\nspec: [" },
    });
    await user.click(screen.getByRole("button", { name: "Apply YAML" }));
    expect(screen.getByLabelText("Authored YAML")).toHaveValue(
      "kind: Workflow\nspec: [",
    );
    expect(screen.getByLabelText("Planner")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "New draft" }));
    await user.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(screen.getByLabelText("Authored YAML")).toHaveValue(
      "kind: Workflow\nspec: [",
    );
    await user.click(screen.getByRole("button", { name: "Revert YAML edits" }));
    expect(screen.getByLabelText("Planner")).not.toBeDisabled();
  });
  it("moves nodes with the keyboard without changing YAML or creating undo history", async () => {
    mount();
    const move = screen.getByRole("button", { name: "Move start" });
    const node = move.closest(".studio-node")!;
    expect(node).toHaveStyle({ left: "370px" });
    fireEvent.keyDown(move, { key: "ArrowRight" });
    expect(node).toHaveStyle({ left: "390px" });
    Object.defineProperty(
      screen.getByLabelText("Scrollable graph canvas"),
      "scrollTo",
      { value: vi.fn() },
    );
    fireEvent.click(screen.getByRole("button", { name: "Auto arrange" }));
    expect(node).toHaveStyle({ left: "370px" });
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
  });
  it("reports an unreachable added stage and restores the source with undo", async () => {
    const user = userEvent.setup();
    mount();
    await user.click(screen.getByRole("button", { name: "+ Stage" }));
    expect(screen.getByText(/Stage is unreachable/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() =>
      expect(
        screen.queryByText(/Stage is unreachable/),
      ).not.toBeInTheDocument(),
    );
    await user.click(screen.getByRole("button", { name: "Redo" }));
    expect(screen.getByText(/Stage is unreachable/)).toBeInTheDocument();
  });
});
