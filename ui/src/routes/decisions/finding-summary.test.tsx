import { screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { FindingSummary } from "./finding-summary";
import { finding } from "../../test/builders";
import { renderWithAPI } from "../../test/render";

it("keeps the source suggestion separate from the current analyst assessment", async () => {
  const value = finding(
    "project",
    "shared",
    {
      state: "confirmed",
      analystVerdict: "true_positive",
      analystSeverity: "critical",
      actorId: "analyst",
      assessedAt: "2026-11-01T10:00:00Z",
      rationale: "Verified against the saved trace.",
    },
    {
      severity_suggestion: "low",
      preconditions: ["The caller has an account."],
      locations: [{ file: "orders.py", line: 42 }],
    },
  );
  renderWithAPI(<FindingSummary finding={value} />, {
    server: () => {
      throw new Error("No read is required by this projection");
    },
  });
  expect(screen.getByText("Critical")).toBeVisible();
  expect(screen.getByText("Source severity suggestion: Low")).toBeVisible();
  expect(
    await screen.findByText("Verified against the saved trace."),
  ).toBeVisible();
  expect(screen.getByText("The caller has an account.")).toBeVisible();
  expect(screen.getByText("orders.py:42")).toBeVisible();
});
it("shows an unassessed canonical Finding without inventing verification", () => {
  renderWithAPI(<FindingSummary finding={finding("project", "unassessed")} />, {
    server: () => {
      throw new Error("No read expected");
    },
  });
  expect(screen.getByText("Not assessed")).toBeVisible();
  expect(screen.queryByText("Supported")).toBeNull();
  expect(
    screen.queryByRole("heading", { name: "Current assessment" }),
  ).toBeNull();
});
