import { useParams, useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { useDocumentTitle } from "../../app/document-title";
import { PaneLayout } from "../../ui";
import { auditProfileLabel } from "../projects/audits/labels";
import { NoReportSelected, ReportDetail } from "./detail";
import { filterSearch, readFilters } from "./filters";
import { ReportsList } from "./list";
import { useReportCheck } from "./report-data";

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
  useDocumentTitle(
    auditId === undefined || check.data === undefined
      ? "Reports"
      : `${auditProfileLabel(check.data)} report`,
  );
  return (
    <PaneLayout
      listLabel="Reports"
      detailLabel="Report"
      showDetail={auditId !== undefined}
      backLink={{
        to: `/reports${filterSearch(filters)}`,
        label: "Back to reports",
      }}
      list={<ReportsList selectedId={auditId} filters={filters} />}
      detail={
        auditId === undefined ? (
          <NoReportSelected />
        ) : (
          <ReportDetail key={auditId} auditId={auditId} check={check} />
        )
      }
    />
  );
}
