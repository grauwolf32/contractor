import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { FilterChips, ListRow, ListSection } from "./list";
import { StatusGlyph } from "./status";

describe("ListSection", () => {
  it("labels the list with its heading and shows count and aside", () => {
    render(
      <ListSection title="Decide" count={1} aside="only you can confirm">
        <ListRow title="IDOR on service reports" />
      </ListSection>,
    );
    const section = screen.getByRole("region", { name: "Decide" });
    expect(
      within(section).getByRole("heading", { level: 2, name: "Decide" }),
    ).toBeInTheDocument();
    expect(within(section).getByText("1")).toBeInTheDocument();
    expect(within(section).getByText("only you can confirm")).toBeVisible();
    expect(within(section).getAllByRole("listitem")).toHaveLength(1);
  });

  it("renders a plain list without a title", () => {
    render(
      <ListSection>
        <ListRow title="crapi-workshop" />
      </ListSection>,
    );
    expect(screen.queryByRole("region")).toBeNull();
    expect(screen.getByRole("list")).toBeInTheDocument();
  });

  it("names the list and declares its keys on the list itself", () => {
    const { rerender } = render(
      <ListSection
        aria-label="Experiments"
        aria-keyshortcuts="J K ArrowDown ArrowUp Enter"
      >
        <ListRow title="Trace instructions" />
      </ListSection>,
    );
    const list = screen.getByRole("list", { name: "Experiments" });
    expect(list).toHaveAttribute(
      "aria-keyshortcuts",
      "J K ArrowDown ArrowUp Enter",
    );
    rerender(
      <ListSection title="Endpoints" aria-keyshortcuts="J K">
        <ListRow title="GET /orders" />
      </ListSection>,
    );
    const section = screen.getByRole("region", { name: "Endpoints" });
    expect(within(section).getByRole("list")).toHaveAttribute(
      "aria-keyshortcuts",
      "J K",
    );
    expect(section).not.toHaveAttribute("aria-keyshortcuts");
    rerender(
      <ListSection>
        <ListRow title="GET /orders" />
      </ListSection>,
    );
    expect(screen.getByRole("list")).not.toHaveAttribute("aria-keyshortcuts");
    expect(screen.getByRole("list")).not.toHaveAttribute("aria-label");
  });
});

describe("ListRow", () => {
  it("marks only the selected link with aria-current", () => {
    render(
      <MemoryRouter>
        <ListSection title="Projects">
          <ListRow to="/projects/a" title="crapi-workshop" selected />
          <ListRow to="/projects/b" title="crapi-identity" />
        </ListSection>
      </MemoryRouter>,
    );
    const selected = screen.getByRole("link", { name: "crapi-workshop" });
    expect(selected).toHaveAttribute("aria-current", "true");
    expect(selected).toHaveAttribute("href", "/projects/a");
    expect(
      screen.getByRole("link", { name: "crapi-identity" }),
    ).not.toHaveAttribute("aria-current");
  });

  it("keeps inline actions outside the title link", async () => {
    const onRetry = vi.fn();
    render(
      <MemoryRouter>
        <ul>
          <ListRow
            to="/checks/1"
            title="One endpoint couldn't be checked"
            glyph={<StatusGlyph tone="blocked" />}
            meta={["crapi-workshop", null, "AI response length limit"]}
          >
            <button type="button" onClick={onRetry}>
              Retry
            </button>
          </ListRow>
        </ul>
      </MemoryRouter>,
    );
    const link = screen.getByRole("link", {
      name: "One endpoint couldn't be checked",
    });
    const retry = screen.getByRole("button", { name: "Retry" });
    expect(link).not.toContainElement(retry);
    await userEvent.setup().click(retry);
    expect(onRetry).toHaveBeenCalledTimes(1);
    const item = screen.getByRole("listitem");
    expect(item).toHaveTextContent("crapi-workshop·AI response length limit");
    for (const separator of item.querySelectorAll(".ui-row-sep")) {
      expect(separator).toHaveAttribute("aria-hidden", "true");
    }
  });

  it("also calls onSelect when a linked row is clicked", async () => {
    const onSelect = vi.fn();
    render(
      <MemoryRouter>
        <ul>
          <ListRow to="/issues/a/1" title="IDOR" onSelect={onSelect} />
        </ul>
      </MemoryRouter>,
    );
    await userEvent.setup().click(screen.getByRole("link", { name: "IDOR" }));
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("declares the row's keys on its link or button", () => {
    render(
      <MemoryRouter>
        <ul>
          <ListRow
            to="/checks?check=a"
            title="WSTG 4.2"
            ariaKeyShortcuts="J K ArrowDown ArrowUp Enter"
          />
          <ListRow
            title="All activity"
            onSelect={() => {}}
            ariaKeyShortcuts="J K"
          />
          <ListRow title="Not selectable" ariaKeyShortcuts="J K" />
          <ListRow to="/checks?check=b" title="Without keys" />
        </ul>
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: "WSTG 4.2" })).toHaveAttribute(
      "aria-keyshortcuts",
      "J K ArrowDown ArrowUp Enter",
    );
    expect(
      screen.getByRole("button", { name: "All activity" }),
    ).toHaveAttribute("aria-keyshortcuts", "J K");
    // A row without a link or button has nothing to focus, so no keys.
    const [, , plain] = screen.getAllByRole("listitem");
    expect(plain!.querySelector("[aria-keyshortcuts]")).toBeNull();
    expect(
      screen.getByRole("link", { name: "Without keys" }),
    ).not.toHaveAttribute("aria-keyshortcuts");
  });

  it("selects through a button when there is no URL", async () => {
    const onSelect = vi.fn();
    const { rerender } = render(
      <ul>
        <ListRow title="All activity" onSelect={onSelect} />
      </ul>,
    );
    const button = screen.getByRole("button", { name: "All activity" });
    expect(button).not.toHaveAttribute("aria-current");
    await userEvent.setup().click(button);
    expect(onSelect).toHaveBeenCalledTimes(1);
    rerender(
      <ul>
        <ListRow title="All activity" onSelect={onSelect} selected />
      </ul>,
    );
    expect(button).toHaveAttribute("aria-current", "true");
  });
});

describe("FilterChips", () => {
  it("presses the current option and reports the chosen one", async () => {
    const onChange = vi.fn();
    render(
      <FilterChips
        label="Filter by status"
        value="proposed"
        onChange={onChange}
        options={[
          { value: "proposed", label: "Needs review", count: 1 },
          { value: "confirmed", label: "Confirmed", count: 0 },
          { value: "rejected", label: "Not an issue" },
        ]}
      />,
    );
    const group = screen.getByRole("group", { name: "Filter by status" });
    const pressed = within(group).getByRole("button", {
      name: "Needs review 1",
    });
    expect(pressed).toHaveAttribute("aria-pressed", "true");
    const confirmed = within(group).getByRole("button", {
      name: "Confirmed 0",
    });
    expect(confirmed).toHaveAttribute("aria-pressed", "false");
    await userEvent.setup().click(confirmed);
    expect(onChange).toHaveBeenCalledWith("confirmed");
  });

  it("changes nothing when the pressed chip is pressed again", async () => {
    const onChange = vi.fn();
    render(
      <FilterChips
        label="Pair filter"
        value="all"
        onChange={onChange}
        options={[
          { value: "all", label: "All pairs" },
          { value: "regressions", label: "Regressions" },
        ]}
      />,
    );
    const user = userEvent.setup();
    const pressed = screen.getByRole("button", { name: "All pairs" });
    await user.click(pressed);
    pressed.focus();
    await user.keyboard("{Enter}");
    await user.keyboard(" ");
    expect(onChange).not.toHaveBeenCalled();
    expect(pressed).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Regressions" }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("regressions");
  });
});
