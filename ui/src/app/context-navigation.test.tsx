import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";
import { ContextLink, ReturnLink } from "./context-navigation";

describe("contextual navigation", () => {
  it("returns from a file through its Run to the original filtered list", async () => {
    const router = createMemoryRouter(
      [
        {
          path: "/runs",
          element: (
            <ContextLink to="/runs/example" returnLabel="Completed Runs">
              Open Run
            </ContextLink>
          ),
        },
        {
          path: "/runs/example",
          element: (
            <>
              <ReturnLink to="/runs" label="All Runs" />
              <ContextLink to="/files/report" returnLabel="Run results">
                Open report
              </ContextLink>
            </>
          ),
        },
        {
          path: "/files/report",
          element: <ReturnLink to="/files" label="All files" />,
        },
      ],
      {
        initialEntries: [
          "/runs?view=completed&state=failed&cursor=next&label=purpose%3Dtest",
        ],
      },
    );
    render(<RouterProvider router={router} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("link", { name: "Open Run" }));
    await user.click(screen.getByRole("link", { name: "Open report" }));
    await user.click(screen.getByRole("link", { name: "← Run results" }));
    await user.click(screen.getByRole("link", { name: "← Completed Runs" }));
    expect(router.state.location.pathname + router.state.location.search).toBe(
      "/runs?view=completed&state=failed&cursor=next&label=purpose%3Dtest",
    );
  });
  it("uses the local fallback for an external return destination", () => {
    const router = createMemoryRouter(
      [
        {
          path: "/detail",
          element: <ReturnLink to="/runs" label="All Runs" />,
        },
      ],
      {
        initialEntries: [
          {
            pathname: "/detail",
            state: { returnTo: "//example.com", returnLabel: "External" },
          },
        ],
      },
    );
    render(<RouterProvider router={router} />);
    expect(screen.getByRole("link", { name: "← All Runs" })).toHaveAttribute(
      "href",
      "/runs",
    );
  });
});
