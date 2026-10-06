import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { IdChip, Kbd, MethodChip } from "./chips";
import { shortenId } from "./format";

const CHECK_ID = "0f3c2a1b-5d6e-4f70-8a9b-1c2d3e4f9e7d";

function stubClipboard(value: unknown) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value,
  });
}

afterEach(() => {
  Reflect.deleteProperty(navigator, "clipboard");
  window.getSelection()?.removeAllRanges();
});

function liveRegion(container: HTMLElement): HTMLElement {
  const region = container.querySelector<HTMLElement>('[aria-live="polite"]');
  if (region === null) throw new Error("live region missing");
  return region;
}

describe("IdChip", () => {
  it("shows a short form with the full value on hover", () => {
    const { rerender } = render(<IdChip value={CHECK_ID} label="check ID" />);
    expect(screen.getByText("0f3c2a1b…9e7d")).toHaveAttribute(
      "title",
      CHECK_ID,
    );
    rerender(<IdChip value="9fceb02" label="commit" />);
    expect(screen.getByText("9fceb02")).toBeInTheDocument();
    rerender(
      <IdChip
        value="openapi-operation-trace@3"
        label="check type version"
        display="openapi-operation-trace@3"
      />,
    );
    expect(screen.getByText("openapi-operation-trace@3")).toBeInTheDocument();
    expect(shortenId("1234567890abcdef")).toBe("1234567890abcdef");
    expect(shortenId("1234567890abcdefg")).toBe("12345678…defg");
  });

  it("copies the full value and announces it", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard({ writeText });
    const { container } = render(<IdChip value={CHECK_ID} label="check ID" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy check ID" }));
    expect(writeText).toHaveBeenCalledWith(CHECK_ID);
    expect(await screen.findByText("Copied")).toBe(liveRegion(container));
    expect(container.querySelector(".ui-id-selectable")).toBeNull();
  });

  it("selects the value for Ctrl+C when the clipboard refuses", async () => {
    stubClipboard({
      writeText: vi.fn().mockRejectedValue(new DOMException("denied")),
    });
    const { container } = render(<IdChip value={CHECK_ID} label="check ID" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy check ID" }));
    expect(await screen.findByText("Press Ctrl+C to copy")).toBe(
      liveRegion(container),
    );
    const selectable = container.querySelector(".ui-id-selectable");
    expect(selectable).toHaveTextContent(CHECK_ID);
    expect(selectable).toHaveAttribute("aria-hidden", "true");
    expect(window.getSelection()?.toString()).toBe(CHECK_ID);
  });

  it("falls back the same way without a Clipboard API", async () => {
    stubClipboard(undefined);
    const { container } = render(<IdChip value={CHECK_ID} label="check ID" />);
    fireEvent.click(screen.getByRole("button", { name: "Copy check ID" }));
    expect(await screen.findByText("Press Ctrl+C to copy")).toBe(
      liveRegion(container),
    );
    expect(window.getSelection()?.toString()).toBe(CHECK_ID);
  });

  it("names ⌘+C on Apple platforms", async () => {
    stubClipboard(undefined);
    Object.defineProperty(navigator, "platform", {
      configurable: true,
      value: "MacIntel",
    });
    try {
      const { container } = render(
        <IdChip value={CHECK_ID} label="check ID" />,
      );
      fireEvent.click(screen.getByRole("button", { name: "Copy check ID" }));
      expect(await screen.findByText("Press ⌘+C to copy")).toBe(
        liveRegion(container),
      );
    } finally {
      Reflect.deleteProperty(navigator, "platform");
    }
  });
});

describe("MethodChip and Kbd", () => {
  it("render the method upper-cased and the key as kbd", () => {
    render(
      <p>
        <MethodChip method="get" /> <Kbd>J</Kbd>
      </p>,
    );
    expect(screen.getByText("GET")).toHaveClass("ui-method-chip");
    expect(screen.getByText("J").tagName).toBe("KBD");
  });
});
