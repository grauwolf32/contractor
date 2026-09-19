import type { AuditCoverageRow } from "../../../api/audits";

export function auditCheckTitle(
  row: Pick<AuditCoverageRow, "subjectKey" | "ordinal" | "details">,
): string {
  const operation = row.details?.taskDocument.operation;
  if (operation !== null && typeof operation === "object") {
    const method: unknown = Object.getOwnPropertyDescriptor(
      operation,
      "method",
    )?.value;
    const path: unknown = Object.getOwnPropertyDescriptor(
      operation,
      "path",
    )?.value;
    if (
      typeof method === "string" &&
      /^(get|put|post|delete|options|head|patch|trace)$/i.test(method) &&
      typeof path === "string" &&
      path.startsWith("/") &&
      path.length <= 2048
    )
      return `${method.toUpperCase()} ${path}`;
  }
  return /^op-[a-f0-9]{16,}$/i.test(row.subjectKey)
    ? `Check ${row.ordinal + 1}`
    : row.subjectKey;
}
