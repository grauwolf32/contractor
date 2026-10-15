import { QueryClient } from "@tanstack/react-query";
import { act, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import * as build from "../../test/builders";
import { deferred } from "../../test/deferred";
import { json, page, type Fallback } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { useProjectArtifactInventory } from "./inventory";

type Inventory = ReturnType<typeof useProjectArtifactInventory>;

/** The inventory hook over a fake Server answering `handle`. */
function renderInventory(handle: Fallback): () => Inventory {
  let current: Inventory | undefined;
  function Probe() {
    current = useProjectArtifactInventory("project_example");
    return null;
  }
  renderWithAPI(<Probe />, {
    server: handle,
    queryClient: new QueryClient({
      defaultOptions: { queries: { retry: false } },
    }),
  });
  return () => {
    if (current === undefined) throw new Error("inventory not rendered");
    return current;
  };
}

const artifact = (name: string) =>
  build.conforming("ArtifactMetadata", {
    artifact: { namespace: "sources", name, revision: "r1" },
    mediaType: "application/zip",
    size: 10,
    current: true,
    frozen: false,
    createdAt: "2026-09-01T10:00:00Z",
  });

describe("Project matching inventory", () => {
  it("does not offer a partial sole match while later pages are loading", async () => {
    const second = deferred();
    let laterPage = false;
    const result = renderInventory(async (_request, url) => {
      if (url.searchParams.has("cursor")) {
        laterPage = true;
        await second.promise;
        return json(page([artifact("second.promise")]));
      }
      return json(page([artifact("first")], "next"));
    });
    await waitFor(() => expect(laterPage).toBe(true));
    expect(result().data).toBeUndefined();
    await act(async () => {
      second.resolve();
      await second.promise;
    });
    await waitFor(() => expect(result().data).toHaveLength(2));
  });
  it("rejects a cyclic inventory and supports an explicit fresh retry", async () => {
    let failed = true;
    let reads = 0;
    const result = renderInventory(async () => {
      reads++;
      return json(page([artifact("first")], failed ? "repeated" : undefined));
    });
    await waitFor(() => expect(result().isError).toBe(true));
    expect(result().data).toBeUndefined();
    expect(reads).toBe(2);
    failed = false;
    await act(async () => {
      await result().refetch();
    });
    await waitFor(() => expect(result().data).toHaveLength(1));
  });
});
