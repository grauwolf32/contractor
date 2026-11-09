import type { ProjectFindingSummary } from "../../../api/audits";
import { MAX_EXPORT_SELECTION } from "../../../api/finding-exports";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { Button, TechnicalDetails } from "../../../ui";
import type { useFindingExport } from "./finding-export-state";
import "./finding-export.css";

type ExportState = ReturnType<typeof useFindingExport>;

export function FindingExportPanel({
  state,
  visible,
  readPending,
  stale,
  onRefresh,
}: {
  state: ExportState;
  visible: readonly ProjectFindingSummary[];
  readPending: boolean;
  stale: boolean;
  onRefresh: () => void;
}) {
  const error = state.publication.error;
  const conflict = state.conflict;
  const canSelectPage =
    new Set([...state.selected.keys(), ...visible.map((f) => f.findingId)])
      .size <= MAX_EXPORT_SELECTION;
  return (
    <details
      className="issues-export"
      open={state.open}
      onToggle={(event) => state.setOpen(event.currentTarget.open)}
    >
      <summary>Export SARIF</summary>
      <div className="ui-stack" data-gap="3">
        <p className="ui-caption">
          Select possible issues from this check. Selections can span pages, up
          to {MAX_EXPORT_SELECTION} possible issues.
        </p>
        <div className="ui-toolbar">
          <Button
            size="xs"
            disabled={
              state.busy ||
              readPending ||
              stale ||
              conflict ||
              visible.length === 0 ||
              !canSelectPage
            }
            onClick={() => state.selectVisible(visible)}
          >
            Select this page
          </Button>
          <Button
            size="xs"
            disabled={
              state.busy || stale || conflict || state.selected.size === 0
            }
            onClick={state.clear}
          >
            Clear selection
          </Button>
          <span role="status">{state.selected.size} selected for export</span>
        </div>
        <label className="issues-export-option">
          <input
            type="checkbox"
            checked={state.includeDuplicates}
            disabled={state.busy || stale || conflict}
            onChange={(event) =>
              state.changeDuplicates(event.currentTarget.checked)
            }
          />{" "}
          Include duplicates
        </label>
        <p className="ui-caption">
          By default, one selected possible issue is exported from each
          duplicate group. Every selected issue and its own review stay in the
          saved snapshot.
        </p>
        {stale || conflict ? (
          <div className="issues-notice" data-tone="warning" role="alert">
            <p>
              This check or a selected possible issue changed. Clear the
              selection and refresh before creating an export.
            </p>
            <Button
              size="xs"
              disabled={state.busy}
              onClick={() => {
                state.clear();
                onRefresh();
              }}
            >
              Clear selection and refresh
            </Button>
          </div>
        ) : null}
        {error === null ? null : (
          <ErrorNotice
            error={error}
            context="Could not create the SARIF export"
          />
        )}
        {state.published === undefined ? (
          <Button
            size="sm"
            disabled={
              state.selected.size === 0 ||
              (!state.canRetry && (readPending || stale)) ||
              conflict ||
              state.download.isPending
            }
            pending={state.publication.isPending}
            pendingLabel="Creating export…"
            onClick={state.canRetry ? state.retry : state.publish}
          >
            {error === null ? "Create SARIF export" : "Retry export"}
          </Button>
        ) : (
          <div className="ui-stack" data-gap="2">
            <p role="status">
              Export ready: {state.published.resultCount}{" "}
              {state.published.resultCount === 1 ? "result" : "results"} from{" "}
              {state.published.selectedCount} selected possible issues.{" "}
              {state.published.warningCount}{" "}
              {state.published.warningCount === 1 ? "warning" : "warnings"}.
            </p>
            <Button
              size="sm"
              pending={state.download.isPending}
              pendingLabel="Downloading…"
              onClick={() => state.download.mutate("content")}
            >
              Download SARIF
            </Button>
            <TechnicalDetails>
              <p className="ui-caption">
                Saved {formatTimestamp(state.published.snapshotAt)}. Downloads
                use this saved export.
              </p>
              <div className="ui-toolbar">
                <Button
                  size="xs"
                  disabled={state.busy}
                  onClick={() => state.download.mutate("snapshot")}
                >
                  Download snapshot
                </Button>
                <Button
                  size="xs"
                  disabled={state.busy}
                  onClick={() => state.download.mutate("diagnostics")}
                >
                  Download diagnostics
                </Button>
              </div>
            </TechnicalDetails>
          </div>
        )}
        {state.download.error === null ? null : (
          <ErrorNotice
            error={state.download.error}
            context="Could not download the saved export"
          />
        )}
      </div>
    </details>
  );
}
