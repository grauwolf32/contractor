import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type RefCallback,
} from "react";

export type ShortcutHandler = (event: KeyboardEvent) => void;

/**
 * Key name → handler. Names are lowercase `KeyboardEvent.key` values ("j",
 * "?", "enter", "escape", "arrowdown", "space") or "mod+<key>" for Ctrl on
 * Windows and Linux and ⌘ on macOS ("mod+k", "mod+enter").
 */
export type ShortcutBindings = Readonly<Record<string, ShortcutHandler>>;

export interface ShortcutOptions {
  /** Default true. */
  enabled?: boolean | undefined;
  /**
   * Keep the bindings working while a modal dialog (an element with
   * aria-modal="true") is open, for handlers that belong to that dialog. They
   * then only react to keys pressed inside a dialog.
   */
  allowInDialog?: boolean | undefined;
}

const KEY_ALIASES: Readonly<Record<string, string>> = {
  " ": "space",
  Spacebar: "space",
  Esc: "escape",
  Up: "arrowup",
  Down: "arrowdown",
  Left: "arrowleft",
  Right: "arrowright",
};

/** Keys that focused controls use themselves (activation, caret, scroll). */
const NAVIGATION_KEYS = new Set([
  "enter",
  "space",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
  "home",
  "end",
  "pageup",
  "pagedown",
]);

/** Input types that take no typed text; letters pressed on them are free. */
const NON_TEXT_INPUT_TYPES = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "hidden",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);

const EDITABLE_SELECTOR = '[contenteditable]:not([contenteditable="false"])';

const INTERACTIVE_SELECTOR = [
  "a[href]",
  "button",
  "input",
  "select",
  "textarea",
  "summary",
  EDITABLE_SELECTOR,
  ...[
    "button",
    "link",
    "checkbox",
    "radio",
    "switch",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "slider",
    "spinbutton",
    "combobox",
    "listbox",
    "textbox",
    "tree",
    "treegrid",
    "grid",
  ].map((role) => `[role="${role}"]`),
].join(",");

const ARROW_WIDGET_SELECTOR = [
  'input[type="radio"]',
  'input[type="range"]',
  ...[
    "radio",
    "radiogroup",
    "slider",
    "spinbutton",
    "listbox",
    "tablist",
    "menu",
    "menubar",
    "tree",
    "treegrid",
    "grid",
    "combobox",
  ].map((role) => `[role="${role}"]`),
].join(",");

/**
 * True when keys pressed on the target type text: text-like inputs,
 * textareas, selects and contenteditable regions. Checkboxes, radios and
 * other inputs that take no text are not typing targets.
 */
export function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target instanceof HTMLInputElement) {
    return !NON_TEXT_INPUT_TYPES.has(target.type);
  }
  if (
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  ) {
    return true;
  }
  return target.closest(EDITABLE_SELECTOR) !== null;
}

function usesKeyItself(target: EventTarget | null, key: string): boolean {
  return (
    NAVIGATION_KEYS.has(key) &&
    target instanceof Element &&
    target.closest(INTERACTIVE_SELECTOR) !== null
  );
}

function handlesArrowKeys(target: EventTarget | null): boolean {
  return (
    target instanceof Element && target.closest(ARROW_WIDGET_SELECTOR) !== null
  );
}

/** True on macOS and iOS, where "mod" is ⌘. */
export function isApplePlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  const hinted = (
    navigator as Navigator & { userAgentData?: { platform?: string } }
  ).userAgentData?.platform;
  return /mac|iphone|ipad|ipod/i.test(hinted ?? navigator.platform);
}

/** Label of the "mod" key for key hints: "⌘" on Apple platforms, else "Ctrl". */
export function modKeyLabel(): string {
  return isApplePlatform() ? "⌘" : "Ctrl";
}

/**
 * Names this event can match. Layouts that type non-Latin letters (for
 * example Cyrillic) also match by the physical key, so J stays J.
 */
function keyNames(event: KeyboardEvent): string[] {
  // Browsers fire keydown without a key for autofill.
  if (typeof event.key !== "string" || event.key === "") return [];
  const name = KEY_ALIASES[event.key] ?? event.key.toLowerCase();
  const physical = /^Key([A-Z])$/.exec(event.code)?.[1]?.toLowerCase();
  if (
    physical !== undefined &&
    physical !== name &&
    name.length === 1 &&
    // Non-ASCII characters only: Latin layouts such as Dvorak keep their own
    // letters.
    name.charCodeAt(0) > 0x7f
  ) {
    return [name, physical];
  }
  return [name];
}

interface ShortcutMatch {
  key: string;
  mod: boolean;
  handler: ShortcutHandler;
}

function findShortcut(
  bindings: ShortcutBindings,
  event: KeyboardEvent,
): ShortcutMatch | undefined {
  const names = keyNames(event);
  const modHeld = event.ctrlKey || event.metaKey;
  for (const [binding, handler] of Object.entries(bindings)) {
    const normalized = binding.trim().toLowerCase();
    const mod = normalized.startsWith("mod+") && normalized.length > 4;
    const key = mod ? normalized.slice(4) : normalized;
    if (!names.includes(key)) continue;
    if (event.altKey || modHeld !== mod) continue;
    return { key, mod, handler };
  }
  return undefined;
}

function blockedByModal(
  target: EventTarget | null,
  allowInDialog: boolean,
): boolean {
  const modals = Array.from(document.querySelectorAll('[aria-modal="true"]'));
  if (modals.length === 0) return false;
  if (!allowInDialog) return true;
  return !(
    target instanceof Node && modals.some((modal) => modal.contains(target))
  );
}

/**
 * Page-level keyboard shortcuts on one document keydown listener.
 *
 * A binding is skipped when the event was already handled
 * (`defaultPrevented`) or is part of an IME composition; when a plain-key
 * binding meets Ctrl, ⌘ or Alt; when focus is in a text field, textarea,
 * select or contenteditable (except "escape" and "mod+…" bindings); when a
 * focused control uses Enter, Space, arrows, Home, End or Page keys itself;
 * and while a modal dialog is open unless `allowInDialog` is set. A binding
 * that fires calls `preventDefault`, so when two hooks bind the same key the
 * first listener wins.
 */
export function useShortcuts(
  bindings: ShortcutBindings,
  options: ShortcutOptions = {},
): void {
  const enabled = options.enabled ?? true;
  const allowInDialog = options.allowInDialog ?? false;
  const latest = useRef(bindings);
  useLayoutEffect(() => {
    latest.current = bindings;
  });

  useEffect(() => {
    if (!enabled) return undefined;
    function onKeyDown(event: KeyboardEvent) {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.key === "Process"
      ) {
        return;
      }
      const match = findShortcut(latest.current, event);
      if (match === undefined) return;
      const { target } = event;
      if (
        !match.mod &&
        match.key !== "escape" &&
        (isTextEntryTarget(target) || usesKeyItself(target, match.key))
      ) {
        return;
      }
      if (blockedByModal(target, allowInDialog)) return;
      event.preventDefault();
      match.handler(event);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [enabled, allowInDialog]);
}

export interface ListNavigationOptions {
  /** Number of items in the list. */
  count: number;
  /** Selected item, or -1 when nothing is selected. */
  index: number;
  /** Selects another item (put the selection in the URL). */
  onMove: (index: number) => void;
  /** Opens the selected item on Enter. Without it Enter keeps its native action. */
  onOpen?: ((index: number) => void) | undefined;
  /** Default true. */
  enabled?: boolean | undefined;
}

export interface ListNavigationContainerProps {
  ref: RefCallback<HTMLElement>;
  onKeyDown: (event: ReactKeyboardEvent<HTMLElement>) => void;
}

export interface ListNavigation {
  /** Spread on the element that contains the list rows. */
  containerProps: ListNavigationContainerProps;
}

/**
 * J / K anywhere on the page (through useShortcuts) and ↓ / ↑ / Home / End /
 * Enter while focus is inside the list container move or open the
 * selection. Moves clamp at both ends and skip `onMove` when nothing would
 * change; with nothing selected, J and K select the first item.
 *
 * After a move the hook scrolls the selected row (the element with
 * aria-current="true" inside the container, as ListRow renders it) into view
 * and, when focus was inside the list, moves focus to it. Enter on the
 * selected row, or on the container itself, calls `onOpen`; Enter on other
 * links and buttons keeps their own action.
 */
export function useListNavigation({
  count,
  index,
  onMove,
  onOpen,
  enabled = true,
}: ListNavigationOptions): ListNavigation {
  const container = useRef<HTMLElement | null>(null);
  // The move the hook asked for, revealed once the selection arrives there.
  const reveal = useRef<{ index: number; focus: boolean } | null>(null);
  const latest = useRef({ count, index, onMove, onOpen, enabled });
  useLayoutEffect(() => {
    latest.current = { count, index, onMove, onOpen, enabled };
  });

  const moveTo = useCallback((target: number) => {
    const state = latest.current;
    if (state.count <= 0) return;
    const next = Math.min(Math.max(target, 0), state.count - 1);
    if (next === state.index) return;
    const root = container.current;
    reveal.current = {
      index: next,
      focus: root !== null && root.contains(document.activeElement),
    };
    state.onMove(next);
  }, []);

  const step = useCallback(
    (delta: 1 | -1) => {
      const current = latest.current.index;
      moveTo(current < 0 ? 0 : current + delta);
    },
    [moveTo],
  );

  useShortcuts(
    { j: () => step(1), k: () => step(-1) },
    { enabled: enabled && count > 0 },
  );

  useEffect(() => {
    const pending = reveal.current;
    const root = container.current;
    if (pending === null || pending.index !== index || root === null) return;
    reveal.current = null;
    const selected = root.querySelector<HTMLElement>('[aria-current="true"]');
    if (selected === null) return;
    if (pending.focus) selected.focus();
    if (typeof selected.scrollIntoView === "function") {
      selected.scrollIntoView({ block: "nearest" });
    }
  }, [index]);

  const ref = useCallback((node: HTMLElement | null) => {
    container.current = node;
  }, []);

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLElement>) => {
      const state = latest.current;
      if (
        !state.enabled ||
        event.defaultPrevented ||
        event.nativeEvent.isComposing ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey
      ) {
        return;
      }
      const { target } = event;
      if (isTextEntryTarget(target)) return;
      switch (event.key) {
        case "ArrowDown":
        case "ArrowUp":
          if (handlesArrowKeys(target)) return;
          event.preventDefault();
          step(event.key === "ArrowDown" ? 1 : -1);
          return;
        case "Home":
        case "End":
          if (handlesArrowKeys(target)) return;
          event.preventDefault();
          moveTo(event.key === "Home" ? 0 : state.count - 1);
          return;
        case "Enter": {
          if (state.onOpen === undefined || state.index < 0) return;
          const onSelectedRow =
            target instanceof Element &&
            target.closest('[aria-current="true"]') !== null;
          const onOtherControl =
            target instanceof Element &&
            target !== event.currentTarget &&
            target.closest(INTERACTIVE_SELECTOR) !== null;
          if (onOtherControl && !onSelectedRow) return;
          event.preventDefault();
          state.onOpen(state.index);
          return;
        }
        default:
          return;
      }
    },
    [moveTo, step],
  );

  return { containerProps: { ref, onKeyDown } };
}
