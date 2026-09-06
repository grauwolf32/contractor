import {
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button",
  "input:not([type='hidden'])",
  "select",
  "textarea",
  "[contenteditable='true']",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

interface DialogLayer {
  container: HTMLDivElement;
  panel: HTMLElement;
  initialFocus: () => HTMLElement | null;
  requestClose: () => void;
  returnFocus: HTMLElement | null;
}

interface ElementState {
  ariaHidden: string | null;
  inert: boolean;
}

const layers: DialogLayer[] = [];
const inactiveElements = new Map<HTMLElement, ElementState>();
let bodyOverflow: string | undefined;

function isDisabled(element: HTMLElement): boolean {
  return element.matches(":disabled") || element.closest("[inert]") !== null;
}

function focusableElements(root: ParentNode): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter(
    (element) =>
      element.isConnected &&
      !isDisabled(element) &&
      element.closest("[hidden], [aria-hidden='true']") === null,
  );
}

function focusLayer(layer: DialogLayer): void {
  if (layers.at(-1) !== layer || !layer.panel.isConnected) return;
  const requested = layer.initialFocus();
  if (
    requested !== null &&
    requested.isConnected &&
    layer.panel.contains(requested) &&
    !isDisabled(requested)
  ) {
    requested.focus();
    return;
  }
  (focusableElements(layer.panel)[0] ?? layer.panel).focus();
}

function restoreElement(element: HTMLElement): void {
  const saved = inactiveElements.get(element);
  if (saved === undefined) return;
  if (saved.inert) element.setAttribute("inert", "");
  else element.removeAttribute("inert");
  if (saved.ariaHidden === null) element.removeAttribute("aria-hidden");
  else element.setAttribute("aria-hidden", saved.ariaHidden);
  inactiveElements.delete(element);
}

function makeElementInactive(element: HTMLElement): void {
  if (!inactiveElements.has(element)) {
    inactiveElements.set(element, {
      ariaHidden: element.getAttribute("aria-hidden"),
      inert: element.hasAttribute("inert"),
    });
  }
  element.setAttribute("inert", "");
  element.setAttribute("aria-hidden", "true");
}

function reconcileInactiveElements(): void {
  const top = layers.at(-1);
  const shouldBeInactive = new Set<HTMLElement>();
  if (top !== undefined) {
    for (const child of document.body.children) {
      if (child instanceof HTMLElement && child !== top.container) {
        shouldBeInactive.add(child);
      }
    }
  }
  for (const element of inactiveElements.keys()) {
    if (!shouldBeInactive.has(element)) restoreElement(element);
  }
  for (const element of shouldBeInactive) makeElementInactive(element);
}

function restoreFocus(layer: DialogLayer): void {
  const target = layer.returnFocus;
  if (
    target !== null &&
    target.isConnected &&
    !isDisabled(target) &&
    target.closest("[hidden], [aria-hidden='true']") === null
  ) {
    target.focus();
    return;
  }
  const top = layers.at(-1);
  if (top !== undefined) {
    focusLayer(top);
    return;
  }
  focusableElements(document.body)[0]?.focus();
}

function onDocumentKeyDown(event: KeyboardEvent): void {
  const top = layers.at(-1);
  if (top === undefined) return;
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopImmediatePropagation();
    top.requestClose();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = focusableElements(top.panel);
  if (focusable.length === 0) {
    event.preventDefault();
    top.panel.focus();
    return;
  }
  const first = focusable[0];
  const last = focusable.at(-1);
  const active = document.activeElement;
  if (!top.panel.contains(active)) {
    event.preventDefault();
    first?.focus();
  } else if (event.shiftKey && (active === first || active === top.panel)) {
    event.preventDefault();
    last?.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first?.focus();
  }
}

function onDocumentFocusIn(event: FocusEvent): void {
  const top = layers.at(-1);
  if (
    top === undefined ||
    (event.target instanceof Node && top.panel.contains(event.target))
  ) {
    return;
  }
  event.stopImmediatePropagation();
  focusLayer(top);
}

function registerLayer(layer: DialogLayer): () => void {
  if (layers.length === 0) {
    bodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onDocumentKeyDown, true);
    document.addEventListener("focusin", onDocumentFocusIn, true);
  }
  layers.push(layer);
  reconcileInactiveElements();
  focusLayer(layer);

  return () => {
    const index = layers.lastIndexOf(layer);
    if (index >= 0) layers.splice(index, 1);
    layer.container.remove();
    reconcileInactiveElements();
    if (layers.length === 0) {
      document.removeEventListener("keydown", onDocumentKeyDown, true);
      document.removeEventListener("focusin", onDocumentFocusIn, true);
      document.body.style.overflow = bodyOverflow ?? "";
      bodyOverflow = undefined;
    }
    queueMicrotask(() => restoreFocus(layer));
  };
}

export interface DialogProps {
  children: ReactNode;
  className: string;
  labelledBy: string;
  onRequestClose: () => void;
  describedBy?: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
  role?: "alertdialog" | "dialog";
}

export function Dialog({
  children,
  className,
  labelledBy,
  onRequestClose,
  describedBy,
  initialFocusRef,
  role = "dialog",
}: DialogProps) {
  const [container] = useState(() => {
    const element = document.createElement("div");
    element.dataset.contractorDialogLayer = "";
    return element;
  });
  const panel = useRef<HTMLElement>(null);
  const closeCallback = useRef(onRequestClose);

  useLayoutEffect(() => {
    closeCallback.current = onRequestClose;
  }, [onRequestClose]);

  useLayoutEffect(() => {
    document.body.append(container);
    const currentPanel = panel.current;
    if (currentPanel === null) {
      container.remove();
      throw new Error("Dialog panel was not mounted");
    }
    const active = document.activeElement;
    const unregister = registerLayer({
      container,
      panel: currentPanel,
      initialFocus: () => initialFocusRef?.current ?? null,
      requestClose: () => closeCallback.current(),
      returnFocus: active instanceof HTMLElement ? active : null,
    });
    return unregister;
  }, [container, initialFocusRef]);

  return createPortal(
    <div
      className="project-dialog-backdrop"
      role="presentation"
      onClick={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
      onSubmit={(event) => event.stopPropagation()}
    >
      <section
        ref={panel}
        className={className}
        role={role}
        aria-modal="true"
        aria-labelledby={labelledBy}
        aria-describedby={describedBy}
        tabIndex={-1}
      >
        {children}
      </section>
    </div>,
    container,
  );
}
