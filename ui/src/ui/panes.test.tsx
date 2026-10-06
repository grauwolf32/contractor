import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ListRow } from "./list";
import { DetailHeader, DetailPane, ListPane, PaneLayout } from "./panes";
import { StatusChip } from "./status";

describe("PaneLayout", () => {
  it("renders a labelled list and detail section", () => {
    const { container } = render(
      <MemoryRouter>
        <PaneLayout
          listLabel="Possible issues"
          detailLabel="Review"
          showDetail={false}
          list={<p>List</p>}
          detail={<p>Detail</p>}
        />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("region", { name: "Possible issues" }),
    ).toHaveTextContent("List");
    expect(screen.getByRole("region", { name: "Review" })).toHaveTextContent(
      "Detail",
    );
    expect(container.firstElementChild).toHaveAttribute("data-show", "list");
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("offers the back link with the detail on one-pane screens", () => {
    const { container } = render(
      <MemoryRouter>
        <PaneLayout
          listLabel="Possible issues"
          detailLabel="Review"
          showDetail
          backLink={{ to: "/issues", label: "Back to issues" }}
          list={<p>List</p>}
          detail={<p>Detail</p>}
        />
      </MemoryRouter>,
    );
    expect(container.firstElementChild).toHaveAttribute("data-show", "detail");
    const back = within(
      screen.getByRole("region", { name: "Review" }),
    ).getByRole("link", { name: "Back to issues" });
    expect(back).toHaveAttribute("href", "/issues");
  });
});

describe("PaneLayout on one-pane screens", () => {
  // jsdom has no media queries: apply the ≤ 820 px rule directly.
  let style: HTMLStyleElement;
  beforeEach(() => {
    style = document.createElement("style");
    style.textContent = `
      .ui-panes[data-show="detail"] > .ui-panes-list,
      .ui-panes[data-show="list"] > .ui-panes-detail { display: none; }`;
    document.head.append(style);
  });
  afterEach(() => style.remove());

  /** Like a page: the selection is the URL, and Back clears it. */
  function Layout({
    selected,
    showDetail = selected !== undefined,
    marksSelection = true,
    rows = ["crapi-workshop", "vampi", "crapi-identity"],
  }: {
    selected?: string;
    showDetail?: boolean;
    marksSelection?: boolean;
    rows?: string[];
  }) {
    return (
      <MemoryRouter>
        <PaneLayout
          listLabel="Checks"
          detailLabel="Selected check"
          showDetail={showDetail}
          backLink={{ to: "/checks", label: "Back to checks" }}
          list={
            <ul>
              {rows.map((name) => (
                <ListRow
                  key={name}
                  to={`/checks?check=${name}`}
                  title={name}
                  selected={marksSelection && name === selected}
                >
                  <button type="button">Open the run of {name}</button>
                </ListRow>
              ))}
            </ul>
          }
          detail={<p>Detail of {selected}</p>}
        />
      </MemoryRouter>
    );
  }

  it("moves focus to the detail, and back to the row it came from", () => {
    const { rerender } = render(<Layout />);
    // A tap on a phone does not focus the link: focus stays on the body.
    rerender(<Layout selected="vampi" />);
    expect(
      screen.getByRole("region", { name: "Selected check" }),
    ).toHaveFocus();
    // Back clears the selection; focus still returns to the row.
    rerender(<Layout />);
    expect(screen.getByRole("link", { name: "vampi" })).toHaveFocus();
  });

  it("returns to the selected row when the selection stays", () => {
    const { rerender } = render(<Layout selected="vampi" showDetail={false} />);
    screen.getByRole("link", { name: "vampi" }).focus();
    rerender(<Layout selected="vampi" />);
    rerender(<Layout selected="vampi" showDetail={false} />);
    expect(screen.getByRole("link", { name: "vampi" })).toHaveFocus();
  });

  it("follows J / K moves made while the detail shows", () => {
    const { rerender } = render(<Layout />);
    rerender(<Layout selected="crapi-workshop" />);
    rerender(<Layout selected="vampi" />);
    rerender(<Layout selected="crapi-identity" />);
    rerender(<Layout />);
    expect(screen.getByRole("link", { name: "crapi-identity" })).toHaveFocus();
  });

  it("returns to what had focus when the list marks no selection", () => {
    const { rerender } = render(<Layout marksSelection={false} />);
    const action = screen.getByRole("button", {
      name: "Open the run of vampi",
    });
    action.focus();
    rerender(<Layout marksSelection={false} selected="vampi" />);
    expect(
      screen.getByRole("region", { name: "Selected check" }),
    ).toHaveFocus();
    rerender(<Layout marksSelection={false} />);
    expect(action).toHaveFocus();
  });

  it("focuses the list when the row is gone", () => {
    const { rerender } = render(<Layout />);
    rerender(<Layout selected="vampi" />);
    rerender(<Layout rows={["crapi-workshop", "crapi-identity"]} />);
    expect(screen.getByRole("region", { name: "Checks" })).toHaveFocus();
  });

  it("leaves focus alone when it is outside the hidden pane", () => {
    const { rerender } = render(
      <>
        <button type="button">Command palette</button>
        <Layout />
      </>,
    );
    const outside = screen.getByRole("button", { name: "Command palette" });
    outside.focus();
    rerender(
      <>
        <button type="button">Command palette</button>
        <Layout selected="vampi" />
      </>,
    );
    expect(outside).toHaveFocus();
  });
});

describe("ListPane", () => {
  it("renders the title block, toolbar, body and footer", () => {
    render(
      <ListPane
        aria-label="Inbox"
        title="Inbox"
        subtitle="2 need you"
        actions={<button type="button">Everything</button>}
        toolbar={<p>Toolbar</p>}
        footer={<p>Keys</p>}
      >
        <p>Rows</p>
      </ListPane>,
    );
    const pane = screen.getByRole("region", { name: "Inbox" });
    expect(
      within(pane).getByRole("heading", { level: 1, name: "Inbox" }),
    ).toBeInTheDocument();
    expect(pane).toHaveTextContent(
      /2 need you.*Everything.*Toolbar.*Rows.*Keys/,
    );
  });

  it("can take a custom header and a lower title level", () => {
    const { rerender } = render(
      <ListPane title="Projects" titleAs="h2">
        <p>Rows</p>
      </ListPane>,
    );
    expect(
      screen.getByRole("heading", { level: 2, name: "Projects" }),
    ).toBeInTheDocument();
    rerender(
      <ListPane header={<p>Custom header</p>} title="Ignored">
        <p>Rows</p>
      </ListPane>,
    );
    expect(screen.getByText("Custom header")).toBeInTheDocument();
    expect(screen.queryByText("Ignored")).toBeNull();
  });
});

describe("DetailHeader and DetailPane", () => {
  it("renders breadcrumb, title, status, meta and footer", () => {
    render(
      <MemoryRouter>
        <DetailPane
          aria-label="Selected endpoint"
          header={
            <DetailHeader
              breadcrumb={[
                { label: "Checks", to: "/checks" },
                { label: "crapi-workshop" },
              ]}
              title="API endpoint trace"
              status={<StatusChip tone="progress">Running</StatusChip>}
              meta="Started Oct 4, 23:51"
              actions={<button type="button">Pause</button>}
            />
          }
          footer={<p>Decision</p>}
        >
          <p>Body</p>
        </DetailPane>
      </MemoryRouter>,
    );
    const pane = screen.getByRole("region", { name: "Selected endpoint" });
    const breadcrumb = within(pane).getByRole("navigation", {
      name: "Breadcrumb",
    });
    expect(
      within(breadcrumb).getByRole("link", { name: "Checks" }),
    ).toHaveAttribute("href", "/checks");
    expect(within(breadcrumb).getByText("crapi-workshop")).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(
      within(pane).getByRole("heading", {
        level: 2,
        name: "API endpoint trace",
      }),
    ).toBeInTheDocument();
    expect(pane).toHaveTextContent(
      /Running.*Pause.*Started Oct 4, 23:51.*Body.*Decision/,
    );
  });

  it("can use an h1 title without a breadcrumb", () => {
    render(<DetailHeader title="crapi-workshop" titleAs="h1" />);
    expect(
      screen.getByRole("heading", { level: 1, name: "crapi-workshop" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("navigation")).toBeNull();
  });
});
