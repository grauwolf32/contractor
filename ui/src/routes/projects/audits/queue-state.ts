import { useSearchParams } from "react-router";
import type { AuditFindingPage, AuditReviewPage } from "../../../api/audits";

export function useAuditQueue() {
  const [params, setParams] = useSearchParams();
  function change(key: string, value: string) {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    next.delete("auditRevision");
    if (value === "" || value === "all") next.delete(key);
    else next.set(key, value);
    setParams(next, { preventScrollReset: true });
  }
  function refresh() {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    next.delete("auditRevision");
    setParams(next, { replace: true, preventScrollReset: true });
  }
  const revision = params.get("auditRevision");
  return {
    params,
    change,
    refresh,
    request: {
      ...(params.has("cursor") ? { cursor: params.get("cursor")! } : {}),
      ...(revision !== null &&
      /^[1-9][0-9]*$/.test(revision) &&
      Number.isSafeInteger(Number(revision))
        ? { auditRevision: Number(revision) }
        : {}),
    },
    next(page: AuditFindingPage | AuditReviewPage) {
      if (page.page.nextCursor === undefined) return;
      const next = new URLSearchParams(params);
      next.set("cursor", page.page.nextCursor);
      next.set("auditRevision", String(page.auditRevision));
      setParams(next, { preventScrollReset: true });
    },
  };
}
