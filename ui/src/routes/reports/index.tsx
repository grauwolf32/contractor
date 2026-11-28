import { useParams, useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { useDocumentTitle } from "../../app/document-title";
import { PaneLayout } from "../../ui";
import { auditProfileLabel } from "../../modules/checks/labels";
import { NoReportSelected, ReportDetail } from "./detail";
import { filterSearch, readFilters } from "./filters";
import { ReportsList } from "./list";
import { useReportCheck } from "../../modules/checks/reports";

/**
 * /reports and /reports/:auditId: reports of every project in a list pane,
 * the selected report (the path) in the detail pane. The list's filters live
 * in the query string and carry over when a report is selected.
 */
export function ReportsRoute() {
  const api = usePublicAPI();
  const { auditId } = useParams();
  const [params] = useSearchParams();
  const filters = readFilters(params);
  const check = useReportCheck(api, auditId);
  const outsideProject =
    filters.project !== null &&
    check.data !== undefined &&
    check.data.projectId !== filters.project;
  const selectedId = outsideProject ? undefined : auditId;
  useDocumentTitle(
    selectedId === undefined || check.data === undefined
      ? "Reports"
      : `${auditProfileLabel(check.data)} report`,
  );
  return (
    <PaneLayout
      listLabel="Reports"
      detailLabel="Report"
      showDetail={selectedId !== undefined}
      backLink={{
        to: `/reports${filterSearch(filters)}`,
        label: "Back to reports",
      }}
      list={<ReportsList selectedId={selectedId} filters={filters} />}
      detail={
        selectedId === undefined ? (
          <NoReportSelected />
        ) : (
          <ReportDetail key={selectedId} auditId={selectedId} check={check} />
        )
      }
    />
  );
}
