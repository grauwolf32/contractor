import type { Audit } from "../../../api/audits";
import { useURLCursorStack } from "../../../app/pagination";
import {
  useResearchHypotheses,
  useResearchMaterials,
  useResearchTasks,
} from "../../../modules/checks/research";

/** Route-owned URL state; query identity and polling belong to modules/checks. */
export function useTaskPage(audit: Audit, hypothesis?: string) {
  const pages = useURLCursorStack({
    param: hypothesis ? "historyCursor" : "tasksCursor",
    clearOnChange: hypothesis ? ["historyTask", "task"] : ["task"],
  });
  const query = useResearchTasks(audit, pages.cursor, hypothesis);
  return { query, pages };
}

export function useHypothesisPage(audit: Audit, enabled = true) {
  const pages = useURLCursorStack({
    param: "hypothesesCursor",
    clearOnChange: ["hypothesis", "historyTask", "historyCursor"],
  });
  const query = useResearchHypotheses(audit, pages.cursor, enabled);
  return { query, pages };
}

export function useMaterialPage(audit: Audit, enabled = true) {
  const pages = useURLCursorStack({ param: "materialsCursor" });
  const query = useResearchMaterials(audit, pages.cursor, enabled);
  return { query, pages };
}
