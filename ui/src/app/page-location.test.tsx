import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import type { Location } from "react-router";
import { createMemoryRouter, MemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";

import { usePageLocationNow } from "./page-location";

type Read = () => Location | undefined;

function Page({ onRead }: { onRead: (read: Read) => void }) {
  const read = usePageLocationNow();
  useEffect(() => onRead(read), [onRead, read]);
  return <p>Page</p>;
}

function gate() {
  let release = () => {};
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { wait, release };
}

describe("usePageLocationNow", () => {
  it("follows the router while the page stays, and lets go once it is left", async () => {
    let read: Read = () => undefined;
    const capture = (next: Read) => {
      read = next;
    };
    const slow = gate();
    const router = createMemoryRouter(
      [
        { path: "/", element: <Page onRead={capture} /> },
        {
          path: "/elsewhere",
          lazy: async () => {
            await slow.wait;
            return { element: <p>Elsewhere</p> };
          },
        },
      ],
      { initialEntries: ["/?item=a"] },
    );
    render(<RouterProvider router={router} />);
    await screen.findByText("Page");
    expect(read()?.search).toBe("?item=a");

    // Another item of the same page: still here, at the new address.
    await act(() => router.navigate("/?item=b"));
    expect(read()?.search).toBe("?item=b");

    // Leaving: the next page's code is still loading and this page is still
    // on screen, but the user is on the way out.
    act(() => {
      void router.navigate("/elsewhere");
    });
    await waitFor(() => expect(router.state.navigation.state).toBe("loading"));
    expect(screen.getByText("Page")).toBeInTheDocument();
    const leaving = read;
    expect(leaving()).toBeUndefined();

    await act(async () => {
      slow.release();
      await slow.wait;
    });
    await screen.findByText("Elsewhere");
    expect(leaving()).toBeUndefined();
  });

  it("uses the rendered location outside a data router", () => {
    let read: Read = () => undefined;
    render(
      <MemoryRouter initialEntries={["/checks/new?project=p"]}>
        <Page
          onRead={(next) => {
            read = next;
          }}
        />
      </MemoryRouter>,
    );
    expect(read()?.pathname).toBe("/checks/new");
    expect(read()?.search).toBe("?project=p");
  });
});
