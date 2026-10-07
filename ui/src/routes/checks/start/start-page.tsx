import { useNavigate } from "react-router";

import {
  DetailPane,
  EmptyState,
  PaneLayout,
  useListNavigation,
} from "../../../ui";
import { SetupBody, SetupFooter, SetupHeader } from "./setup";
import { TypeList } from "./type-list";
import { useStartCheck } from "./use-start-check";

/**
 * Start a check on one project: check types on the left, grouped by
 * readiness; the chosen type's setup on the right with a pinned
 * "Start check" bar.
 */
export function StartCheckPage({
  projectId,
  initialObjective,
}: {
  projectId: string;
  initialObjective: string;
}) {
  const navigate = useNavigate();
  const model = useStartCheck(projectId, initialObjective);
  const { ordered, selection } = model;
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
    <PaneLayout
      listLabel="Check types"
      detailLabel="Set up the check"
      showDetail={model.typeChosen}
      backLink={{
        to: model.href({ type: undefined }),
        label: "Back to check types",
      }}
      list={<TypeList model={model} containerProps={containerProps} />}
      detail={
        <DetailPane
          header={<SetupHeader model={model} />}
          footer={
            model.listReady && selection !== undefined ? (
              <SetupFooter model={model} />
            ) : undefined
          }
        >
          {!model.listReady ? (
            model.catalog.query.data === undefined &&
            model.catalog.query.error !== null ? (
              <EmptyState title="Check types could not be loaded">
                Try again from the list.
              </EmptyState>
            ) : (
              <p className="start-quiet" role="status">
                Loading check types…
              </p>
            )
          ) : selection === undefined ? (
            <EmptyState title="No check type to set up">
              This server lists no check types yet.
            </EmptyState>
          ) : (
            <SetupBody model={model} />
          )}
        </DetailPane>
      }
    />
  );
}
