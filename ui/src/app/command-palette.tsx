import { useQuery } from "@tanstack/react-query";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { useLocation, useNavigate } from "react-router";

import { listAuditPresets } from "../api/audit-presets";
import type { AuditProfile } from "../api/audits";
import type { PublicAPI } from "../api/client";
import { usePublicAPI } from "../api/context";
import {
  type CrossProjectCheck,
  useAllChecks,
  useProjectsIndex,
} from "../api/cross-project";
import type { Project } from "../api/projects";
import { queryKeys } from "../api/query-keys";
import { listWorkflows, type WorkflowSummary } from "../api/workflows";
import { useSession } from "../auth/session";
import { Kbd, modKeyLabel, useShortcuts } from "../ui";
import { type Destination, destinationsFor, projectIdOf } from "./destinations";
import { Dialog } from "./dialog";
import { Icon, type IconName } from "./icon";
import { capitalize, checkStateLabel, TERMS } from "./vocabulary";

/** Results each group shows for a search. */
export const RESULTS_PER_GROUP = 6;

// Catalog data changes rarely; reopening the palette reuses it for a while.
const CATALOG_STALE_MS = 60_000;

interface PaletteOption {
  readonly key: string;
  readonly label: string;
  readonly meta?: string | undefined;
  readonly icon: IconName;
  readonly to: string;
  /** The label, normalized for matching anywhere in it. */
  readonly labelText: string;
  /** Words of the label and keywords, matched from their start. */
  readonly words: readonly string[];
}

interface PaletteGroup {
  readonly id: string;
  readonly title: string;
  readonly options: readonly PaletteOption[];
  /** Shown before anything is typed. */
  readonly suggested: boolean;
}

interface ListedOption {
  readonly option: PaletteOption;
  /** Position in the whole list, for aria-activedescendant. */
  readonly index: number;
}

interface ListedGroup {
  readonly id: string;
  readonly title: string;
  readonly options: readonly ListedOption[];
}

function normalize(text: string): string {
  return text.toLocaleLowerCase("en-US").replace(/\s+/g, " ").trim();
}

/** Whole words and their parts: "openapi-from-source@2" and "source". */
function wordsOf(text: string): string[] {
  const words = normalize(text).split(" ");
  return [
    ...new Set([
      ...words,
      ...words.flatMap((word) => word.split(/[^\p{L}\p{N}]+/u)),
    ]),
  ].filter((word) => word !== "");
}

/**
 * An option. A search matches anywhere in its label, and at the start of a
 * word of its label or `keywords` (former names, project names, IDs).
 */
function option(
  value: Omit<PaletteOption, "labelText" | "words">,
  keywords = "",
): PaletteOption {
  return {
    ...value,
    labelText: normalize(value.label),
    words: wordsOf(`${value.label} ${keywords}`),
  };
}

function withQuery(path: string, query: Record<string, string>): string {
  const search = new URLSearchParams(query).toString();
  return search === "" ? path : `${path}?${search}`;
}

function actionOptions(project: {
  id: string | undefined;
  name: string | undefined;
}): PaletteOption[] {
  return [
    option(
      {
        key: "action:start",
        label: "Start a check",
        meta: project.name === undefined ? undefined : `In ${project.name}`,
        icon: "checks",
        to: withQuery(
          "/checks/new",
          project.id === undefined ? {} : { project: project.id },
        ),
      },
      "new audit run",
    ),
    option(
      {
        key: "action:new-project",
        label: "New project",
        icon: "plus",
        to: "/projects?new=1",
      },
      "create add",
    ),
    option(
      {
        key: "action:settings",
        label: "Settings",
        icon: "settings",
        to: "/operations/settings",
      },
      "account git ssh key preferences",
    ),
  ];
}

// The Library's Files (the personal artifact list) has no rail item of its
// own; the palette keeps it one jump away.
const LIBRARY_FILES = option(
  {
    key: "go:files",
    label: capitalize(TERMS.files),
    meta: TERMS.library,
    icon: "artifacts",
    to: "/artifacts",
  },
  "library artifacts uploads",
);

function destinationOptions(
  destinations: readonly Destination[],
): PaletteOption[] {
  return destinations.flatMap((destination) => {
    const entry = option(
      {
        key: `go:${destination.id}`,
        label: destination.label,
        icon: destination.icon,
        to: destination.to,
      },
      destination.keywords,
    );
    return destination.id === "library" ? [entry, LIBRARY_FILES] : [entry];
  });
}

function projectOptions(projects: readonly Project[]): PaletteOption[] {
  return projects.map((project) =>
    option(
      {
        key: `project:${project.projectId}`,
        label: project.name,
        meta: project.description.split("\n", 1)[0]?.trim() || undefined,
        icon: "projects",
        to: `/projects/${encodeURIComponent(project.projectId)}`,
      },
      `${project.description} ${project.projectId}`,
    ),
  );
}

function checkOptions(checks: readonly CrossProjectCheck[]): PaletteOption[] {
  return checks.map(({ project, audit }) => {
    const objective = audit.scope.objective?.trim();
    const state = checkStateLabel(audit.state).label;
    return option(
      {
        key: `check:${audit.auditId}`,
        label: objective || audit.profile.name,
        meta: `${project.name} · ${state}`,
        icon: "checks",
        to: `/projects/${encodeURIComponent(project.projectId)}/audits/${encodeURIComponent(audit.auditId)}`,
      },
      `${project.name} ${state} ${audit.profile.name} ${audit.auditId}`,
    );
  });
}

/** Newest version first; numeric parts compare as numbers ("10" after "9"). */
function newerVersion(left: string, right: string): number {
  return right.localeCompare(left, "en", { numeric: true });
}

/** One entry per name: its newest version, in name order. */
function newestByName<T>(
  items: readonly T[],
  ref: (item: T) => { name: string; version: string },
): T[] {
  const newest = new Map<string, T>();
  for (const item of items) {
    const { name, version } = ref(item);
    const known = newest.get(name);
    if (known === undefined || newerVersion(version, ref(known).version) < 0)
      newest.set(name, item);
  }
  return [...newest.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, item]) => item);
}

function checkTypeOptions(
  profiles: readonly AuditProfile[],
  projectId: string | undefined,
): PaletteOption[] {
  return newestByName(profiles, (profile) => profile.ref).map((profile) =>
    option(
      {
        key: `check-type:${profile.ref.name}`,
        label: profile.ref.name,
        meta: "Start a check",
        icon: "checklist",
        to: withQuery("/checks/new", {
          ...(projectId === undefined ? {} : { project: projectId }),
          type: profile.ref.name,
        }),
      },
      `${profile.ref.version} ${profile.mode.replaceAll("-", " ")}`,
    ),
  );
}

function workflowOptions(
  workflows: readonly WorkflowSummary[],
): PaletteOption[] {
  return newestByName(workflows, (workflow) => workflow.ref).map((workflow) => {
    const { name, version } = workflow.ref;
    const displayName = workflow.presentation?.displayName.trim();
    return option(
      {
        key: `workflow:${name}`,
        label: displayName || name,
        meta: `${name}@${version}`,
        icon: "catalog",
        to: `/catalog/workflows/${encodeURIComponent(name)}/${encodeURIComponent(version)}`,
      },
      `${name}@${version} ${workflow.presentation?.description ?? ""}`,
    );
  });
}

/**
 * Options of one group that match every word of the query, at most
 * RESULTS_PER_GROUP. Labels that start with the query come first, then
 * labels with a word that starts with its first word, then the rest, each
 * in list order. `query` is normalized and not empty.
 */
function search(
  options: readonly PaletteOption[],
  query: string,
): PaletteOption[] {
  const terms = query.split(" ");
  const first = terms[0] ?? query;
  return options
    .flatMap((candidate, position) => {
      const matches = terms.every(
        (term) =>
          candidate.labelText.includes(term) ||
          candidate.words.some((word) => word.startsWith(term)),
      );
      if (!matches) return [];
      const label = candidate.labelText;
      const rank = label.startsWith(query)
        ? 0
        : wordsOf(label).some((word) => word.startsWith(first))
          ? 1
          : 2;
      return [{ candidate, rank, position }];
    })
    .sort(
      (left, right) => left.rank - right.rank || left.position - right.position,
    )
    .slice(0, RESULTS_PER_GROUP)
    .map(({ candidate }) => candidate);
}

async function workflowInventory(
  api: PublicAPI,
  signal: AbortSignal,
): Promise<WorkflowSummary[]> {
  // The same complete inventory the Library reads under this key.
  const items: WorkflowSummary[] = [];
  const cursors = new Set<string>();
  let cursor: string | undefined;
  do {
    const page = await listWorkflows(api, {
      ...(cursor === undefined ? {} : { cursor }),
      signal,
    });
    items.push(...page.items);
    if (!page.page.hasMore) return items;
    cursor = page.page.nextCursor;
    if (!cursor || cursors.has(cursor))
      throw new Error(
        "Workflow inventory could not be completed. Refresh to retry.",
      );
    cursors.add(cursor);
  } while (!signal.aborted);
  signal.throwIfAborted();
  return items;
}

function PaletteDialog({ onClose }: { onClose: () => void }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { session } = useSession();
  const capabilities = session?.principal.capabilities;
  const [input, setInput] = useState("");
  const [activeKey, setActiveKey] = useState<string>();
  const field = useRef<HTMLInputElement>(null);
  const baseId = useId();
  const titleId = `${baseId}-title`;
  const listboxId = `${baseId}-results`;
  const optionId = (index: number) => `${baseId}-option-${index}`;

  // The dialog is mounted only while the palette is open, so these reads
  // run (and the check pages poll) only then. Project and check pages are
  // shared with the Inbox badge; check types and workflows with the
  // Library.
  const projects = useProjectsIndex();
  const checks = useAllChecks();
  const checkTypes = useQuery({
    queryKey: queryKeys.catalog.auditPresets,
    queryFn: ({ signal }) => listAuditPresets(api, signal),
    staleTime: CATALOG_STALE_MS,
  });
  const workflows = useQuery({
    queryKey: queryKeys.workflows.inventory,
    queryFn: ({ signal }) => workflowInventory(api, signal),
    staleTime: CATALOG_STALE_MS,
  });

  useShortcuts({ "mod+k": onClose }, { allowInDialog: true });

  const projectId = projectIdOf(pathname);
  const projectName = projects.projects.find(
    (project) => project.projectId === projectId,
  )?.name;

  const groups = useMemo<PaletteGroup[]>(
    () => [
      {
        id: "actions",
        title: "Actions",
        options: actionOptions({ id: projectId, name: projectName }),
        suggested: true,
      },
      {
        id: "go",
        title: "Go to",
        options: destinationOptions(destinationsFor(capabilities)),
        suggested: true,
      },
      {
        id: "projects",
        title: "Projects",
        options: projectOptions(projects.projects),
        suggested: false,
      },
      {
        id: "checks",
        title: capitalize(TERMS.checks),
        options: checkOptions(checks.checks),
        suggested: false,
      },
      {
        id: "check-types",
        title: capitalize(TERMS.checkTypes),
        options: checkTypeOptions(checkTypes.data ?? [], projectId),
        suggested: false,
      },
      {
        id: "workflows",
        title: "Workflows",
        options: workflowOptions(workflows.data ?? []),
        suggested: false,
      },
    ],
    [
      capabilities,
      checkTypes.data,
      checks.checks,
      projectId,
      projectName,
      projects.projects,
      workflows.data,
    ],
  );

  const query = normalize(input);
  const listed = useMemo(() => {
    let index = 0;
    const result: ListedGroup[] = [];
    for (const group of groups) {
      const options =
        query === ""
          ? group.suggested
            ? group.options
            : []
          : search(group.options, query);
      if (options.length === 0) continue;
      result.push({
        id: group.id,
        title: group.title,
        options: options.map((candidate) => ({
          option: candidate,
          index: index++,
        })),
      });
    }
    return result;
  }, [groups, query]);
  const flat = useMemo(
    () => listed.flatMap((group) => group.options),
    [listed],
  );
  // The first option is active until the user moves; a typed character
  // starts again from the top.
  const activeIndex = Math.max(
    flat.findIndex(({ option: entry }) => entry.key === activeKey),
    flat.length === 0 ? -1 : 0,
  );
  const active = flat[activeIndex];
  const activeId = active === undefined ? undefined : optionId(active.index);
  const activeOptionKey = active?.option.key;

  useEffect(() => {
    if (activeId === undefined) return;
    const element = document.getElementById(activeId);
    if (typeof element?.scrollIntoView === "function") {
      element.scrollIntoView({ block: "nearest" });
    }
  }, [activeId, activeOptionKey]);

  const loading =
    query !== "" &&
    (projects.isPending ||
      checks.isPending ||
      checkTypes.isPending ||
      workflows.isPending);
  const failed =
    query !== "" &&
    (projects.error !== null ||
      checks.partial ||
      checkTypes.isError ||
      workflows.isError);

  function open(target: PaletteOption) {
    onClose();
    void navigate(target.to);
  }

  function move(step: 1 | -1) {
    if (flat.length === 0) return;
    const next = (activeIndex + step + flat.length) % flat.length;
    setActiveKey(flat[next]?.option.key);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      move(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Enter" && active !== undefined) {
      event.preventDefault();
      open(active.option);
    }
  }

  const resultCount = `${flat.length} ${flat.length === 1 ? "result" : "results"}`;
  const status =
    flat.length > 0
      ? query === ""
        ? ""
        : loading
          ? `${resultCount}, more loading`
          : resultCount
      : loading
        ? "Searching…"
        : "No matches";

  return (
    <Dialog
      className="shell-palette"
      backdropClassName="shell-palette-backdrop"
      labelledBy={titleId}
      initialFocusRef={field}
      onRequestClose={onClose}
      dismissOnBackdrop
    >
      <h2 id={titleId} className="ui-visually-hidden">
        Search or start a check
      </h2>
      <div className="shell-palette-search">
        <Icon name="search" />
        <input
          ref={field}
          className="shell-palette-input"
          type="text"
          role="combobox"
          aria-labelledby={titleId}
          aria-expanded={flat.length > 0}
          aria-controls={listboxId}
          aria-activedescendant={activeId}
          aria-autocomplete="list"
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          placeholder="Search or start a check…"
          value={input}
          onChange={(event) => {
            setInput(event.target.value);
            setActiveKey(undefined);
          }}
          onKeyDown={onKeyDown}
        />
        {/* The Esc key hint, and a Close button on touch screens. */}
        <button
          className="shell-palette-close"
          type="button"
          aria-label="Close"
          aria-keyshortcuts="Escape"
          onClick={onClose}
        >
          <span className="shell-palette-close-key" aria-hidden="true">
            <Kbd>Esc</Kbd>
          </span>
          <span className="shell-palette-close-text" aria-hidden="true">
            Close
          </span>
        </button>
      </div>
      {flat.length > 0 ? (
        <div
          id={listboxId}
          className="shell-palette-results"
          role="listbox"
          aria-label="Results"
        >
          {listed.map((group) => (
            <div
              key={group.id}
              className="shell-palette-group"
              role="group"
              aria-labelledby={`${baseId}-${group.id}`}
            >
              <div
                id={`${baseId}-${group.id}`}
                className="shell-palette-heading"
                role="presentation"
              >
                {group.title}
              </div>
              {group.options.map(({ option: entry, index }) => (
                <div
                  key={entry.key}
                  id={optionId(index)}
                  className="shell-palette-option"
                  role="option"
                  aria-selected={index === activeIndex}
                  onMouseMove={() => {
                    if (index !== activeIndex) setActiveKey(entry.key);
                  }}
                  // Keeps focus in the field; the click opens the option.
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => open(entry)}
                >
                  <Icon name={entry.icon} />
                  <span className="shell-palette-label">{entry.label}</span>
                  {entry.meta === undefined ? null : (
                    <span className="shell-palette-meta">{entry.meta}</span>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      ) : null}
      <p
        className={
          flat.length > 0 ? "ui-visually-hidden" : "shell-palette-empty"
        }
        role="status"
      >
        {status}
      </p>
      {failed ? (
        <p className="shell-palette-note">Some results could not be loaded.</p>
      ) : null}
      <p className="shell-palette-hints" aria-hidden="true">
        <span>
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd> move
        </span>
        <span>
          <Kbd>Enter</Kbd> open
        </span>
        <span>
          <Kbd>{modKeyLabel()}</Kbd>
          <Kbd>K</Kbd> close
        </span>
      </p>
    </Dialog>
  );
}

/**
 * The command palette: Ctrl+K or ⌘+K toggles it from anywhere (except over
 * another dialog). It searches actions, destinations, projects, checks,
 * check types and workflows already readable on the client.
 */
export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  useShortcuts({ "mod+k": () => onOpenChange(true) }, { enabled: !open });
  return open ? <PaletteDialog onClose={() => onOpenChange(false)} /> : null;
}
