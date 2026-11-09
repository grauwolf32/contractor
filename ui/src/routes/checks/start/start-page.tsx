import { MaterialsCard } from "./materials";
import { ExecutionParameters } from "./execution-parameters";
import { useState, useSyncExternalStore } from "react";
import { useLocation, useNavigate } from "react-router";

import { EmptyState, useListNavigation } from "../../../ui";
import {
  ObjectiveField,
  SetupBody,
  SetupFooter,
  SetupHeader,
  SetupOptions,
} from "./setup";
import { TypeList } from "./type-list";
import { useStartCheck } from "./use-start-check";
import { WebScanSetup } from "./web-scan-setup";

function subscribeWidth(callback: () => void) {
  window.addEventListener("resize", callback);
  return () => window.removeEventListener("resize", callback);
}
const narrowWidth = () => window.innerWidth <= 820;

/** One setup workspace; on phones the type picker folds into the document flow. */
export function StartCheckPage({
  projectId,
  initialObjective,
}: {
  projectId: string;
  initialObjective: string;
}) {
  const navigate = useNavigate();
  const model = useStartCheck(projectId, initialObjective);
  const narrow = useSyncExternalStore(subscribeWidth, narrowWidth);
  const location = useLocation();
  const [pickerOpen, setPickerOpen] = useState(
    () => !new URLSearchParams(location.search).has("type"),
  );
  const [search, setSearch] = useState("");
  const ordered = model.ordered.filter((row) =>
    `${row.presentation.label} ${row.presentation.description}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );
  const { selection } = model;
  const { containerProps } = useListNavigation({
    count: model.listReady ? ordered.length : 0,
    index: ordered.findIndex((row) => row.selected),
    onMove: (index) => {
      const row = ordered[index];
      if (row !== undefined) void navigate(row.href, { replace: true });
    },
    onOpen: (index) => {
      const row = ordered[index];
      if (row !== undefined) void navigate(row.href);
    },
  });
  return (
    <div className="start-workspace">
      <SetupHeader model={model} />
      <header className="start-page-heading">
        <div>
          <h1>Start a check</h1>
          <p>Choose what to review. Set up the materials and scope.</p>
        </div>
        <span className="start-project-badge">
          {model.project?.name ?? "Project"}
        </span>
      </header>
      <fieldset
        className="start-objective-wrap"
        disabled={model.launch.pending !== undefined}
      >
        <ObjectiveField model={model} />
      </fieldset>
      <div className="start-workbench">
        <details
          className="start-type-picker"
          open={!narrow || pickerOpen}
          onToggle={(event) => {
            if (narrow) setPickerOpen(event.currentTarget.open);
          }}
        >
          <summary>
            Change check type{" "}
            <span>
              {selection?.presentation.label ?? "Choose a check type"}
            </span>
          </summary>
          <section
            aria-label="Check types"
            onClickCapture={(event) => {
              if (
                narrow &&
                event.target instanceof Element &&
                event.target.closest("a[href]")
              )
                setPickerOpen(false);
            }}
          >
            <TypeList
              model={model}
              rows={ordered}
              search={search}
              onSearch={setSearch}
              containerProps={containerProps}
            />
          </section>
        </details>
        <section className="start-setup" aria-label="Set up the check">
          {!model.listReady ? (
            model.catalog.query.data === undefined &&
            model.catalog.query.error !== null ? (
              <EmptyState title="Check types could not be loaded">
                Try again from the list.
              </EmptyState>
            ) : (
              <p className="ui-caption" role="status">
                Loading check types…
              </p>
            )
          ) : selection === undefined ? (
            <EmptyState title="No check type to set up">
              This server lists no check types yet.
            </EmptyState>
          ) : (
            <fieldset
              className="start-setup-fields"
              disabled={model.launch.pending !== undefined}
            >
              {model.webScan !== undefined ? (
                <>
                  <MaterialsCard model={model} />
                  <ExecutionParameters model={model} />
                  <WebScanSetup
                    model={model.webScan}
                    materials={model.materials}
                    busy={model.launch.pending !== undefined}
                  />
                  <SetupOptions model={model} />
                </>
              ) : (
                <SetupBody model={model} />
              )}
            </fieldset>
          )}
        </section>
        {model.listReady && selection !== undefined ? (
          <SetupFooter model={model} />
        ) : null}
      </div>
    </div>
  );
}
