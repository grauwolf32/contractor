import { screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { finding, findingSummary } from "../../test/builders";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { conflict, json, page } from "../../test/responses";
import { DuplicatePicker } from "./duplicate-picker";

it("continues bounded candidate batches, keeps exact IDs available and restarts a stale snapshot", async () => {
  const requests: number[] = [];
  let stale = true;
  const target = findingSummary(
    finding(
      "project_picker",
      "finding_target",
      {},
      { title: "Original issue" },
    ),
  );
  const server = fakeServer({
    "GET /v1/projects/{projectId}/findings": ({ url }) => {
      const cursor = Number(url.searchParams.get("cursor") ?? 0);
      requests.push(cursor);
      if (cursor === 5 && stale) {
        stale = false;
        return conflict();
      }
      return json({
        ...page(
          cursor === 5 ? [target] : [],
          cursor === 5 ? undefined : String(cursor + 1),
        ),
        projectRevision: 1,
      });
    },
  });
  const onChange = vi.fn();
  const view = renderWithAPI(
    <DuplicatePicker
      projectId="project_picker"
      findingId="finding_current"
      value=""
      onChange={onChange}
    />,
    { server },
  );
  await screen.findByRole("button", { name: "Load more candidates" });
  expect(requests).toEqual([0, 1, 2, 3, 4]);
  await view.user.type(
    screen.getByRole("textbox", { name: "Finding ID" }),
    "x",
  );
  expect(onChange).toHaveBeenCalledWith("x");
  await view.user.click(
    screen.getByRole("button", { name: "Load more candidates" }),
  );
  await screen.findByRole("alert");
  await view.user.click(
    screen.getByRole("button", { name: "restart the list" }),
  );
  await waitFor(() =>
    expect(requests).toEqual([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]),
  );
  await view.user.click(
    await screen.findByRole("button", { name: "Load more candidates" }),
  );
  await view.user.click(
    await screen.findByRole("button", { name: "Original issue" }),
  );
  expect(onChange).toHaveBeenLastCalledWith("finding_target");
  expect(
    screen.queryByRole("button", { name: "Load more candidates" }),
  ).toBeNull();
  expect(requests).toEqual([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
});
