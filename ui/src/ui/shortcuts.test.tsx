import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { ListRow, ListSection } from "./list";
import {
  isTextEntryTarget,
  useListNavigation,
  useShortcuts,
  type ShortcutBindings,
  type ShortcutOptions,
} from "./shortcuts";

function Shortcuts({
  bindings,
  options,
}: {
  bindings: ShortcutBindings;
  options?: ShortcutOptions;
}) {
  useShortcuts(bindings, options);
  return null;
}

describe("useShortcuts", () => {
  it("fires a plain-key binding and prevents the default action", () => {
    const onJ = vi.fn();
    render(<Shortcuts bindings={{ j: onJ }} />);
    const notCancelled = fireEvent.keyDown(document.body, { key: "j" });
    expect(onJ).toHaveBeenCalledTimes(1);
    expect(notCancelled).toBe(false);
    fireEvent.keyDown(document.body, { key: "J", shiftKey: true });
    expect(onJ).toHaveBeenCalledTimes(2);
  });

  it("ignores keys typed into text fields but keeps escape and mod bindings", async () => {
    const onJ = vi.fn();
    const onEscape = vi.fn();
    const onPalette = vi.fn();
    render(
      <>
        <Shortcuts
          bindings={{ j: onJ, escape: onEscape, "mod+k": onPalette }}
        />
        <input aria-label="Search" />
        <textarea aria-label="Notes" />
        <select aria-label="Pick">
          <option>One</option>
        </select>
        <div contentEditable aria-label="Editor" role="textbox" />
      </>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByLabelText("Search"));
    await user.keyboard("j");
    expect(screen.getByLabelText("Search")).toHaveValue("j");
    await user.click(screen.getByLabelText("Notes"));
    await user.keyboard("j");
    expect(screen.getByLabelText("Notes")).toHaveValue("j");
    fireEvent.keyDown(screen.getByLabelText("Pick"), { key: "j" });
    fireEvent.keyDown(screen.getByLabelText("Editor"), { key: "j" });
    expect(onJ).not.toHaveBeenCalled();

    await user.click(screen.getByLabelText("Notes"));
    await user.keyboard("{Escape}");
    expect(onEscape).toHaveBeenCalledTimes(1);
    await user.keyboard("{Control>}k{/Control}");
    await user.keyboard("{Meta>}k{/Meta}");
    expect(onPalette).toHaveBeenCalledTimes(2);
  });

  it("treats checkboxes and radios as non-typing targets", () => {
    const onJ = vi.fn();
    render(
      <>
        <Shortcuts bindings={{ j: onJ }} />
        <input type="checkbox" aria-label="Done" />
        <input type="radio" aria-label="High" />
      </>,
    );
    fireEvent.keyDown(screen.getByLabelText("Done"), { key: "j" });
    fireEvent.keyDown(screen.getByLabelText("High"), { key: "j" });
    expect(onJ).toHaveBeenCalledTimes(2);
    expect(isTextEntryTarget(screen.getByLabelText("Done"))).toBe(false);
  });

  it("ignores plain-key bindings while Ctrl, Meta or Alt is held", () => {
    const onJ = vi.fn();
    const onModJ = vi.fn();
    render(<Shortcuts bindings={{ j: onJ, "mod+j": onModJ }} />);
    for (const modifier of ["ctrlKey", "metaKey", "altKey"] as const) {
      fireEvent.keyDown(document.body, { key: "j", [modifier]: true });
    }
    expect(onJ).not.toHaveBeenCalled();
    // Ctrl and ⌘ trigger the mod binding; Alt does not.
    expect(onModJ).toHaveBeenCalledTimes(2);
  });

  it("skips handled events and IME composition", () => {
    const onJ = vi.fn();
    render(<Shortcuts bindings={{ j: onJ }} />);
    const handled = new KeyboardEvent("keydown", {
      key: "j",
      bubbles: true,
      cancelable: true,
    });
    handled.preventDefault();
    document.body.dispatchEvent(handled);
    fireEvent.keyDown(document.body, { key: "j", isComposing: true });
    fireEvent.keyDown(document.body, { key: "Process", code: "KeyJ" });
    expect(onJ).not.toHaveBeenCalled();
  });

  it("stays off while a modal dialog is open unless allowed in dialogs", () => {
    const onPage = vi.fn();
    const onDialog = vi.fn();
    render(
      <>
        <Shortcuts bindings={{ j: onPage }} />
        <Shortcuts
          bindings={{ k: onDialog }}
          options={{ allowInDialog: true }}
        />
        <div role="dialog" aria-modal="true" aria-label="Palette">
          <button type="button">Inside</button>
        </div>
      </>,
    );
    const inside = screen.getByRole("button", { name: "Inside" });
    fireEvent.keyDown(document.body, { key: "j" });
    fireEvent.keyDown(inside, { key: "j" });
    expect(onPage).not.toHaveBeenCalled();
    fireEvent.keyDown(document.body, { key: "k" });
    expect(onDialog).not.toHaveBeenCalled();
    fireEvent.keyDown(inside, { key: "k" });
    expect(onDialog).toHaveBeenCalledTimes(1);
  });

  it("leaves Enter and arrows to focused controls", () => {
    const onEnter = vi.fn();
    const onDown = vi.fn();
    render(
      <>
        <Shortcuts bindings={{ enter: onEnter, arrowdown: onDown }} />
        <button type="button">Save</button>
      </>,
    );
    const button = screen.getByRole("button", { name: "Save" });
    fireEvent.keyDown(button, { key: "Enter" });
    fireEvent.keyDown(button, { key: "ArrowDown" });
    expect(onEnter).not.toHaveBeenCalled();
    expect(onDown).not.toHaveBeenCalled();
    fireEvent.keyDown(document.body, { key: "Enter" });
    fireEvent.keyDown(document.body, { key: "ArrowDown" });
    expect(onEnter).toHaveBeenCalledTimes(1);
    expect(onDown).toHaveBeenCalledTimes(1);
  });

  it("matches the physical key on non-Latin layouts only", () => {
    const onJ = vi.fn();
    const onH = vi.fn();
    render(<Shortcuts bindings={{ j: onJ, h: onH }} />);
    // Cyrillic layout: the J key types "о".
    fireEvent.keyDown(document.body, { key: "о", code: "KeyJ" });
    expect(onJ).toHaveBeenCalledTimes(1);
    // Dvorak: the J key types "h", which is what the user means.
    fireEvent.keyDown(document.body, { key: "h", code: "KeyJ" });
    expect(onJ).toHaveBeenCalledTimes(1);
    expect(onH).toHaveBeenCalledTimes(1);
  });

  it("can be disabled and always calls the latest handler", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = render(
      <Shortcuts bindings={{ j: first }} options={{ enabled: false }} />,
    );
    fireEvent.keyDown(document.body, { key: "j" });
    expect(first).not.toHaveBeenCalled();
    rerender(<Shortcuts bindings={{ j: first }} />);
    fireEvent.keyDown(document.body, { key: "j" });
    rerender(<Shortcuts bindings={{ j: second }} />);
    fireEvent.keyDown(document.body, { key: "j" });
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("lets the first listener win when two hooks bind the same key", () => {
    const first = vi.fn();
    const second = vi.fn();
    render(
      <>
        <Shortcuts bindings={{ j: first }} />
        <Shortcuts bindings={{ j: second }} />
      </>,
    );
    fireEvent.keyDown(document.body, { key: "j" });
    expect(first.mock.calls.length + second.mock.calls.length).toBe(1);
  });

  it("leaves an event alone when the binding declines it", () => {
    const onStart = vi.fn();
    const when = vi.fn(
      (event: KeyboardEvent) =>
        !(event.target instanceof Element && event.target.closest("a[href]")),
    );
    render(
      <>
        <Shortcuts bindings={{ "mod+enter": { handler: onStart, when } }} />
        <a href="/projects">Change project</a>
        <button type="button">Save</button>
      </>,
    );
    const link = screen.getByRole("link", { name: "Change project" });
    // Declined: no preventDefault, so Ctrl/⌘+Enter opens the link.
    expect(fireEvent.keyDown(link, { key: "Enter", ctrlKey: true })).toBe(true);
    expect(fireEvent.keyDown(link, { key: "Enter", metaKey: true })).toBe(true);
    expect(onStart).not.toHaveBeenCalled();
    expect(when).toHaveBeenCalledTimes(2);
    // Taken: prevented and handled, like the function form.
    const button = screen.getByRole("button", { name: "Save" });
    expect(fireEvent.keyDown(button, { key: "Enter", ctrlKey: true })).toBe(
      false,
    );
    expect(onStart).toHaveBeenCalledTimes(1);
    expect(onStart.mock.calls[0]?.[0]).toBeInstanceOf(KeyboardEvent);
  });

  it("asks a binding only once the other guards let the event through", () => {
    const when = vi.fn(() => true);
    render(
      <>
        <Shortcuts bindings={{ j: { handler: vi.fn(), when } }} />
        <input aria-label="Search" />
      </>,
    );
    fireEvent.keyDown(screen.getByLabelText("Search"), { key: "j" });
    fireEvent.keyDown(document.body, { key: "j", ctrlKey: true });
    expect(when).not.toHaveBeenCalled();
    fireEvent.keyDown(document.body, { key: "j" });
    expect(when).toHaveBeenCalledTimes(1);
  });

  it("passes a declined key on to the next listener", () => {
    const first = vi.fn();
    const second = vi.fn();
    render(
      <>
        <Shortcuts bindings={{ j: { handler: first, when: () => false } }} />
        <Shortcuts bindings={{ j: second }} />
      </>,
    );
    expect(fireEvent.keyDown(document.body, { key: "j" })).toBe(false);
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});

const ITEMS = ["Alpha", "Beta", "Gamma"];

function NavigableList({
  initial = 0,
  onMove,
  onOpen,
}: {
  initial?: number;
  onMove: (index: number) => void;
  onOpen?: (index: number) => void;
}) {
  const [index, setIndex] = useState(initial);
  const { containerProps } = useListNavigation({
    count: ITEMS.length,
    index,
    onMove: (next) => {
      onMove(next);
      setIndex(next);
    },
    onOpen,
  });
  return (
    <>
      <button type="button">Outside</button>
      <div {...containerProps}>
        <ListSection title="Items">
          {ITEMS.map((item, position) => (
            <ListRow
              key={item}
              title={item}
              selected={position === index}
              onSelect={() => setIndex(position)}
            >
              {position === 1 ? <button type="button">Retry</button> : null}
            </ListRow>
          ))}
        </ListSection>
      </div>
    </>
  );
}

function selectedTitle(): string | null {
  return document.querySelector('[aria-current="true"]')?.textContent ?? null;
}

describe("useListNavigation", () => {
  it("moves with J and K and clamps at both ends", async () => {
    const onMove = vi.fn();
    render(<NavigableList onMove={onMove} />);
    const user = userEvent.setup();
    await user.keyboard("k");
    expect(onMove).not.toHaveBeenCalled();
    await user.keyboard("j");
    expect(selectedTitle()).toBe("Beta");
    await user.keyboard("jj");
    expect(selectedTitle()).toBe("Gamma");
    expect(onMove.mock.calls).toEqual([[1], [2]]);
    await user.keyboard("kkk");
    expect(selectedTitle()).toBe("Alpha");
    expect(onMove).toHaveBeenLastCalledWith(0);
  });

  it("starts at the first item when nothing is selected", async () => {
    const onMove = vi.fn();
    render(<NavigableList initial={-1} onMove={onMove} />);
    await userEvent.setup().keyboard("k");
    expect(onMove).toHaveBeenCalledWith(0);
    expect(selectedTitle()).toBe("Alpha");
  });

  it("uses arrows, Home and End only while focus is inside the list", async () => {
    const onMove = vi.fn();
    render(<NavigableList onMove={onMove} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Outside" }));
    await user.keyboard("{ArrowDown}");
    expect(onMove).not.toHaveBeenCalled();

    screen.getByRole("button", { name: "Alpha" }).focus();
    await user.keyboard("{ArrowDown}");
    expect(selectedTitle()).toBe("Beta");
    // Focus follows the selection.
    expect(screen.getByRole("button", { name: "Beta" })).toHaveFocus();
    await user.keyboard("{End}");
    expect(screen.getByRole("button", { name: "Gamma" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(screen.getByRole("button", { name: "Alpha" })).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(onMove.mock.calls).toEqual([[1], [2], [0]]);
  });

  it("opens the selected row on Enter but leaves inline actions alone", async () => {
    const onOpen = vi.fn();
    render(<NavigableList initial={1} onMove={vi.fn()} onOpen={onOpen} />);
    const user = userEvent.setup();
    screen.getByRole("button", { name: "Beta" }).focus();
    await user.keyboard("{Enter}");
    expect(onOpen).toHaveBeenCalledWith(1);
    screen.getByRole("button", { name: "Retry" }).focus();
    await user.keyboard("{Enter}");
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it("keeps J and K out of text fields", async () => {
    const onMove = vi.fn();
    render(
      <>
        <NavigableList onMove={onMove} />
        <input aria-label="Filter" />
      </>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByLabelText("Filter"));
    await user.keyboard("jk");
    expect(onMove).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Filter")).toHaveValue("jk");
  });
});
