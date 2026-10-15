import { PublicAPIError } from "../api/error";

/**
 * Why the Server refused a write that raced with another change:
 *
 * - "stale": 412 Precondition Failed. The request named a revision
 *   (If-Match) that is no longer current; someone else changed the resource
 *   first and nothing was overwritten.
 * - "conflict": 409 Conflict. The resource's current state rules the
 *   request out (it moved on, or the same key was used for other content);
 *   nothing was applied or retried.
 *
 * Either way the page reads the resource again and shows its current state
 * before the user tries again; it never retries on its own.
 */
export type WriteConflict = "stale" | "conflict";

/** The kind of write conflict an error reports, or undefined for any other error. */
export function writeConflict(error: unknown): WriteConflict | undefined {
  if (!(error instanceof PublicAPIError)) return undefined;
  if (error.status === 412) return "stale";
  if (error.status === 409) return "conflict";
  return undefined;
}
