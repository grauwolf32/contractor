import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import type { Location } from "react-router";
import { createMemoryRouter, MemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";

import { deferred } from "../test/deferred";
import { useGuardedNavigate, usePageLocationNow } from "./page-location";

type Read = () => Location | undefined;

function Page({ onRead }: { onRead: (read: Read) => void }) {
  const read = usePageLocationNow();
  useEffect(() => onRead(read), [onRead, read]);
  return <p>Page</p>;
}

describe("usePageLocationNow", () => {
  it("follows the router while the page stays, and lets go once it is left", async () => {
    let read: Read = () => undefined;
    const capture = (next: Read) => {
      read = next;
    };
    const slow = deferred();
    const router = createMemoryRouter(
      [
        { path: "/", element: <Page onRead={capture} /> },
        {
          path: "/elsewhere",
          lazy: async () => {
            await slow.promise;
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
      slow.resolve();
      await slow.promise;
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

describe("useGuardedNavigate", () => {
  type Go = (to: string) => Promise<boolean>;

  function Form({ onReady }: { onReady: (go: Go) => void }) {
    const go = useGuardedNavigate();
    useEffect(() => onReady(go), [onReady, go]);
    return <p>Form</p>;
  }

  function routes(onReady: (go: Go) => void) {
    return createMemoryRouter(
      [
        { path: "/form", element: <Form onReady={onReady} /> },
        { path: "/saved", element: <p>Saved</p> },
        { path: "/elsewhere", element: <p>Elsewhere</p> },
      ],
      { initialEntries: ["/form"] },
    );
  }

  it("navigates while the user is still on the page that asked", async () => {
    let go: Go = async () => false;
    const router = routes((next) => {
      go = next;
    });
    render(<RouterProvider router={router} />);
    await screen.findByText("Form");
    let moved = false;
    await act(async () => {
      moved = await go("/saved");
    });
    expect(moved).toBe(true);
    expect(router.state.location.pathname).toBe("/saved");
  });

  it("leaves a user who moved on where they are", async () => {
    let go: Go = async () => false;
    const router = routes((next) => {
      go = next;
    });
    render(<RouterProvider router={router} />);
    await screen.findByText("Form");
    const late = go;
    await act(() => router.navigate("/elsewhere"));
    let moved = true;
    await act(async () => {
      moved = await late("/saved");
    });
    expect(moved).toBe(false);
    expect(router.state.location.pathname).toBe("/elsewhere");
  });
});
