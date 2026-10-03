import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, useLocation } from "react-router";
import { describe, expect, it } from "vitest";

import {
  nextPageCursor,
  useCursorStack,
  useURLCursorStack,
} from "./pagination";

describe("nextPageCursor", () => {
  it("returns a cursor only for pages that have more items", () => {
    expect(nextPageCursor({ hasMore: true, nextCursor: "c1" })).toBe("c1");
    expect(
      nextPageCursor({ hasMore: false, nextCursor: "c1" }),
    ).toBeUndefined();
    expect(nextPageCursor({ hasMore: true, nextCursor: "" })).toBeUndefined();
    expect(nextPageCursor({ hasMore: true, nextCursor: null })).toBeUndefined();
    expect(nextPageCursor(undefined)).toBeUndefined();
  });
});

describe("useCursorStack", () => {
  it("moves forward, back and to the first page", () => {
    const { result } = renderHook(() => useCursorStack());
    expect(result.current.cursor).toBeUndefined();
    expect(result.current.controls({ hasMore: false })).toMatchObject({
      canGoBack: false,
    });
    expect(
      result.current.controls({ hasMore: false }).nextCursor,
    ).toBeUndefined();

    act(() => result.current.controls({ hasMore: true }).onNext("c1"));
    act(() => result.current.controls({ hasMore: true }).onNext("c2"));
    expect(result.current.cursor).toBe("c2");
    expect(result.current.controls({ hasMore: false }).canGoBack).toBe(true);

    act(() => result.current.controls({ hasMore: false }).onBack());
    expect(result.current.cursor).toBe("c1");

    act(() => result.current.reset());
    expect(result.current.cursor).toBeUndefined();
  });
});

describe("useURLCursorStack", () => {
  function wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={["/runs?state=failed&cursor=c1"]}>
        {children}
      </MemoryRouter>
    );
  }

  it("keeps cursors in repeated search parameters beside the filters", () => {
    const { result } = renderHook(
      () => ({ pages: useURLCursorStack(), location: useLocation() }),
      { wrapper },
    );
    expect(result.current.pages.cursor).toBe("c1");

    act(() => result.current.pages.controls(undefined).onNext("c2"));
    expect(result.current.location.search).toBe(
      "?state=failed&cursor=c1&cursor=c2",
    );
    expect(result.current.pages.cursor).toBe("c2");

    act(() => result.current.pages.controls(undefined).onBack());
    expect(result.current.location.search).toBe("?state=failed&cursor=c1");

    act(() => result.current.pages.reset());
    expect(result.current.location.search).toBe("?state=failed");
  });
});
