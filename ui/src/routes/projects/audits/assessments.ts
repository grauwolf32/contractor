/**
 * Result filter of a check's item list, kept in the URL as `result`. The
 * values are the URL contract: progress links and the report use them.
 */
export type Group =
  "all" | "issues" | "uncertain" | "not-tested" | "complete" | "out-of-scope";
