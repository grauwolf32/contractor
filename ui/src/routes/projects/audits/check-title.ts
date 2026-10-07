import type { AuditCoverageRow } from "../../../api/audits";

const HTTP_METHOD = /^(get|put|post|delete|options|head|patch|trace)$/i;

/** The HTTP operation an endpoint item traces. */
export interface ItemOperation {
  /** Upper case: "GET". */
  method: string;
  /** The path template: "/orders/{id}". */
  path: string;
}

/** The operation of an endpoint item, read from its exact task document. */
export function itemOperation(
  row: Pick<AuditCoverageRow, "details">,
): ItemOperation | undefined {
  const operation = row.details?.taskDocument.operation;
  if (operation === null || typeof operation !== "object") return undefined;
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
    HTTP_METHOD.test(method) &&
    typeof path === "string" &&
    path.startsWith("/") &&
    path.length <= 2048
  )
    return { method: method.toUpperCase(), path };
  return undefined;
}

/** Internal operation keys ("op-<hex>") say nothing to a reader. */
export function isOpaqueSubject(subjectKey: string): boolean {
  return /^op-[a-f0-9]{16,}$/i.test(subjectKey);
}

/**
 * A plain-text name of one item: "GET /orders/{id}" for an endpoint, its
 * subject (requirement key) otherwise, "Item 3" for an opaque key.
 */
export function auditCheckTitle(
  row: Pick<AuditCoverageRow, "subjectKey" | "ordinal" | "details">,
): string {
  const operation = itemOperation(row);
  if (operation !== undefined) return `${operation.method} ${operation.path}`;
  return isOpaqueSubject(row.subjectKey)
    ? `Item ${row.ordinal + 1}`
    : row.subjectKey;
}
