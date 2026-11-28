import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { SourcePreview } from "./source-preview";

describe("numbered source", () => {
  it.each(["one\n\nthree\n", "one\r\n\r\nthree", "one\r\rthree\r", "", "one"])(
    "preserves literal text %j, including blank lines and CRLF",
    async (source) => {
      const { container } = render(<SourcePreview source={source} />);
      expect(container.querySelector("pre")?.textContent).toBe(source);
      await userEvent.click(screen.getByRole("button", { name: "Wrap lines" }));
      expect(container.querySelector("pre")?.dataset.wrap).toBe("false");
      expect(container.querySelector("pre")?.textContent).toBe(source);
    },
  );
  it("bounds newline-heavy payload DOM without dropping text", () => {
    const source = "line\n".repeat(4001);
    const { container } = render(<SourcePreview source={source} />);
    expect(container.querySelector("pre")?.textContent).toBe(source);
    expect(container.querySelectorAll(".materials-source-line")).toHaveLength(
      0,
    );
    expect(screen.getByText(/Line numbers are hidden/)).toBeVisible();
  });
  it("numbers every line through the documented limit", () => {
    const { container } = render(
      <SourcePreview source={"line\n".repeat(4000)} />,
    );
    expect(container.querySelectorAll("[data-line]")).toHaveLength(4000);
    expect(container.querySelector('[data-line="4000"]')).toHaveTextContent(
      "line",
    );
  });
});
