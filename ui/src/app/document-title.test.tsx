import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { documentTitle, useDocumentTitle } from "./document-title";

function Page({ title }: { title?: string | null }) {
  useDocumentTitle(title);
  return null;
}

describe("useDocumentTitle", () => {
  it("suffixes the application name and falls back to it when empty", () => {
    const view = render(<Page title="Projects" />);
    expect(document.title).toBe("Projects · Contractor");
    view.rerender(<Page title="  " />);
    expect(document.title).toBe("Contractor");
    view.rerender(<Page />);
    expect(document.title).toBe("Contractor");
    view.rerender(<Page title="crapi-workshop" />);
    expect(document.title).toBe("crapi-workshop · Contractor");
  });

  it("falls back to the application name when the page unmounts", () => {
    const view = render(<Page title="Projects" />);
    expect(document.title).toBe("Projects · Contractor");
    view.unmount();
    expect(document.title).toBe("Contractor");
  });

  it("keeps a title set by a later page when an earlier one unmounts", () => {
    const earlier = render(<Page title="Projects" />);
    const later = render(<Page title="Runs" />);
    expect(document.title).toBe("Runs · Contractor");
    earlier.unmount();
    expect(document.title).toBe("Runs · Contractor");
    later.unmount();
    expect(document.title).toBe("Contractor");
  });

  it("formats titles without rendering", () => {
    expect(documentTitle("Runs")).toBe("Runs · Contractor");
    expect(documentTitle(null)).toBe("Contractor");
  });
});
