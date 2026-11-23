import { expect, test, type Locator, type Page } from "@playwright/test";

import type { components } from "../src/api/generated/public";
import { audit, finding, findingSource, project } from "../src/test/builders";
import { deferred } from "../src/test/deferred";
import { ownerListResponse } from "../src/test/owner-lists";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

type Schemas = components["schemas"];

// A keyboard user who records a decision keeps focus inside the decision bar
// while the Server has not answered: jsdom keeps focus on a control that
// turns disabled, so only a real browser shows focus falling to <body>.

const PROJECT = "project_shop";
const AUDIT = "audit_shop";
const REASON = "Only administrators reach this view.";

function issue(
  findingId: string,
  title: string,
  createdAt: string,
): Schemas["ProjectFinding"] {
  return finding(
    PROJECT,
    findingId,
    { createdAt, updatedAt: createdAt },
    { title },
  );
}
// One Project owner command from the approved Issue workspace. The response
// is held so keyboard focus and duplicate-submission prevention run in Chromium.
async function mockShop(page: Page) {
  const data = {
    projects: [project(PROJECT, { name: "Shop" })],
    audits: [audit(PROJECT, AUDIT, "completed")],
    findings: [
      issue(
        "finding_orders",
        "Any user can read any order",
        "2026-10-05T09:00:00Z",
      ),
      issue(
        "finding_invoices",
        "Invoices leak through a debug endpoint",
        "2026-10-04T09:00:00Z",
      ),
    ],
  };
  const answer = deferred();
  const decisions: unknown[] = [];
  await mockAPI(page, async (route, request, url) => {
    const owned = ownerListResponse(url, data);
    if (owned !== undefined) return fulfillJSON(route, await owned.json());
    const path = url.pathname;
    if (path === "/v1/projects")
      return fulfillJSON(route, {
        items: data.projects,
        page: { hasMore: false },
      });
    if (path === `/v1/projects/${PROJECT}`)
      return fulfillJSON(route, data.projects[0], 200, { ETag: '"1"' });
    const one = /^\/v1\/projects\/[^/]+\/findings\/([^/]+)$/.exec(path);
    if (one !== null) {
      const item = data.findings.find((f) => f.findingId === one[1]);
      if (item === undefined) return fulfillEmptyPageOrNotFound(route);
      if (request.method() === "PATCH") {
        decisions.push(request.postDataJSON());
        await answer.promise;
        const body = request.postDataJSON() as { rationale: string };
        const updated: Schemas["ProjectFinding"] = {
          ...item,
          state: "rejected",
          revision: item.revision + 1,
          analystVerdict: "false_positive",
          analystSeverity: null,
          rationale: body.rationale,
          actorId: "demo",
          assessedAt: "2026-11-01T10:00:00Z",
        };
        data.findings = data.findings.map((f) =>
          f.findingId === updated.findingId ? updated : f,
        );
        return fulfillJSON(route, { finding: updated, replayed: false }, 200, {
          ETag: `"${updated.revision}"`,
        });
      }
      return fulfillJSON(route, item, 200, { ETag: `"${item.revision}"` });
    }
    const source = data.findings.find(
      (f) =>
        path === `/v1/projects/${PROJECT}/finding-sources/${f.firstReceiptId}`,
    );
    if (source !== undefined) return fulfillJSON(route, findingSource(source));
    if (path.endsWith("/sources") || path.endsWith("/history"))
      return fulfillJSON(route, {
        items: [],
        projectRevision: 1,
        page: { hasMore: false },
      });
    return fulfillEmptyPageOrNotFound(route);
  });
  return { decisions, answer: () => answer.resolve() };
}

/**
 * Records, from now on, every element that lost focus to the page body for
 * longer than the script that blurred it: a `focusout` with no element
 * taking focus, and still no focused element once that script has run. (A
 * browser may blur a control while React locks it, and the commit's layout
 * effects focus another before any task or frame can see the page body.)
 */
async function watchLostFocus(page: Page): Promise<() => Promise<string[]>> {
  await page.evaluate(() => {
    const lost: string[] = [];
    Object.assign(window, { lostFocus: lost });
    document.addEventListener(
      "focusout",
      (event) => {
        const { target } = event;
        if (event.relatedTarget !== null || !(target instanceof Element))
          return;
        queueMicrotask(() => {
          const active = document.activeElement;
          if (active === null || active === document.body)
            lost.push(target.outerHTML.slice(0, 120));
        });
      },
      true,
    );
  });
  return () =>
    page.evaluate(() => [
      ...(window as unknown as { lostFocus: string[] }).lostFocus,
    ]);
}

/**
 * Lets the browser render a few frames and run the tasks they queue, among
 * them the one in which Chromium drops focus from a control inside a
 * container that turned inert (a control that turns disabled loses it at
 * once).
 */
async function settle(page: Page) {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let frames = 3;
        const frame = () => {
          frames -= 1;
          if (frames === 0) setTimeout(resolve, 0);
          else requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
      }),
  );
}

const IN_BAR = "in the decision bar";

/** IN_BAR, or the start tag of the element that has focus instead. */
function focusPlace(bar: Locator): Promise<string> {
  return bar.evaluate((element, inside) => {
    const active = document.activeElement;
    if (active !== null && active !== element && element.contains(active))
      return inside;
    return active === null
      ? "nothing"
      : active.outerHTML.slice(0, active.outerHTML.indexOf(">") + 1);
  }, IN_BAR);
}

interface Submission {
  /** How the decision is sent. */
  name: string;
  /** Sends it; focus is in the reason. */
  send: (page: Page, bar: Locator) => Promise<void>;
  /** Tries to send it again while it records. */
  again: (page: Page, bar: Locator) => Promise<void>;
}

function recordButton(bar: Locator): Locator {
  return bar.getByRole("button", { name: "Record decision" });
}

const SUBMISSIONS: Submission[] = [
  {
    name: "the Record decision button pressed from the keyboard",
    send: async (page, bar) => {
      await page.keyboard.press("Tab");
      await expect(recordButton(bar)).toBeFocused();
      await page.keyboard.press("Enter");
    },
    again: async (page) => {
      await page.keyboard.press("Enter");
      await page.keyboard.press("Space");
    },
  },
  {
    name: "a click on Record decision",
    send: (_page, bar) => recordButton(bar).click(),
    again: (_page, bar) => recordButton(bar).click({ force: true }),
  },
  {
    name: "Ctrl/Cmd+Enter in the reason",
    send: (page) => page.keyboard.press("ControlOrMeta+Enter"),
    again: (page) => page.keyboard.press("ControlOrMeta+Enter"),
  },
  // Safari and Firefox on macOS do not focus a button that is clicked:
  // focus stays on the control the keyboard left it on, which locks.
  {
    name: "a click that leaves focus on a verdict",
    send: async (_page, bar) => {
      await bar.getByRole("button", { name: "Not an issue" }).focus();
      await recordButton(bar).dispatchEvent("click");
    },
    again: (_page, bar) => recordButton(bar).dispatchEvent("click"),
  },
  {
    name: "a click that leaves focus in the slot above the reason",
    send: async (_page, bar) => {
      await bar.getByRole("button", { name: "Use issue summary" }).focus();
      await recordButton(bar).dispatchEvent("click");
    },
    again: (_page, bar) => recordButton(bar).dispatchEvent("click"),
  },
];

for (const submission of SUBMISSIONS) {
  for (const width of [1440, 390]) {
    test(`focus stays in the decision bar while recording, sent with ${submission.name} at ${width}px`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 1000 });
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      const server = await mockShop(page);
      await page.goto("/issues/project_shop/finding_orders");
      const bar = page.getByRole("region", { name: "Your decision" });
      await bar.getByRole("button", { name: "Not an issue" }).click();
      const reason = bar.getByRole("textbox", { name: "Why" });
      await expect(reason).toBeFocused();
      await page.keyboard.type(REASON);
      await expect(recordButton(bar)).toBeEnabled();
      const lostFocus = await watchLostFocus(page);

      await submission.send(page, bar);
      await expect.poll(() => server.decisions.length).toBe(1);
      await expect(bar).toHaveAttribute("aria-busy", "true");
      await expect(bar.getByText("Recording…")).toBeVisible();
      await settle(page);
      expect(await focusPlace(bar)).toBe(IN_BAR);
      expect(await lostFocus()).toEqual([]);

      // Nothing records twice while the first decision is on its way.
      await submission.again(page, bar);
      await settle(page);
      expect(server.decisions).toHaveLength(1);
      expect(await focusPlace(bar)).toBe(IN_BAR);
      expect(await lostFocus()).toEqual([]);

      // Once the Server answers, focus moves on to the next possible issue.
      server.answer();
      await expect(page).toHaveURL(/\/issues\/project_shop\/finding_invoices$/);
      await expect(
        page.locator(".ui-panes-detail .issues-detail-bar"),
      ).toBeFocused();
      expect(server.decisions).toEqual([
        expect.objectContaining({
          verdict: "false_positive",
          rationale: REASON,
        }),
      ]);
      expect(errors).toEqual([]);
    });
  }
}
