import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";

import { collectAuditPages } from "../../api/audit-collections";
import {
  AUDIT_ID_PATTERN,
  listAuditFindings,
  type AuditFinding,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";

export interface DuplicatePickerProps {
  auditId: string;
  /** The possible issue being decided; it cannot be its own original. */
  findingId: string;
  /** The chosen original's ID, or "". */
  value: string;
  onChange: (findingId: string) => void;
  /** Id of the search field, so the caller can move focus to it. */
  searchId: string;
}

function Candidate({
  name,
  id,
  title,
  checked,
  onChange,
}: {
  name: string;
  id: string;
  title: string;
  checked: boolean;
  onChange: (id: string) => void;
}) {
  return (
    <label className="decisions-duplicate-option">
      <input
        type="radio"
        name={name}
        value={id}
        checked={checked}
        onChange={() => onChange(id)}
      />
      <span className="decisions-duplicate-text">
        <span className="decisions-duplicate-title">{title}</span>{" "}
        <span className="decisions-duplicate-id">{id}</span>
      </span>
    </label>
  );
}

/**
 * Chooses the original of a duplicate among the other possible issues of the
 * same check. It reads up to five pages of the check's possible issues and
 * searches them by title or ID; an exact ID from beyond those pages can be
 * used as typed (S19: the Server validates the target).
 */
export function DuplicatePicker({
  auditId,
  findingId,
  value,
  onChange,
  searchId,
}: DuplicatePickerProps) {
  const api = usePublicAPI();
  const name = useId();
  const [query, setQuery] = useState("");
  const candidates = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(auditId),
      "decisions",
      "duplicate-candidates",
    ],
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditFindings(api, auditId, cursor === undefined ? {} : { cursor }),
      ),
  });
  const others: AuditFinding[] = (candidates.data?.items ?? []).filter(
    (candidate) => candidate.findingId !== findingId,
  );
  const needle = query.trim().toLocaleLowerCase();
  const matches =
    needle === ""
      ? others
      : others.filter(
          (candidate) =>
            candidate.findingId.toLocaleLowerCase().includes(needle) ||
            candidate.firstProposal.document.title
              .toLocaleLowerCase()
              .includes(needle),
        );
  const selected = others.find((candidate) => candidate.findingId === value);
  const visible =
    selected === undefined || matches.includes(selected)
      ? matches
      : [selected, ...matches];
  const typed = query.trim();
  const offerTyped =
    candidates.data !== undefined &&
    typed !== "" &&
    typed !== value &&
    typed !== findingId &&
    AUDIT_ID_PATTERN.test(typed) &&
    !others.some((candidate) => candidate.findingId === typed);
  const typedSelected = value !== "" && selected === undefined;
  const truncated = candidates.data?.truncated ?? false;

  return (
    <fieldset className="decisions-duplicate">
      <legend className="decisions-duplicate-legend">Duplicate of</legend>
      <label className="ui-visually-hidden" htmlFor={searchId}>
        Search possible issues in this check
      </label>
      <input
        id={searchId}
        className="decisions-duplicate-search"
        type="search"
        autoComplete="off"
        spellCheck={false}
        placeholder="Search by title or ID"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {visible.length > 0 || typedSelected || offerTyped ? (
        <div className="decisions-duplicate-options">
          {typedSelected ? (
            <Candidate
              name={name}
              id={value}
              title="ID entered by you"
              checked
              onChange={onChange}
            />
          ) : null}
          {visible.map((candidate) => (
            <Candidate
              key={candidate.findingId}
              name={name}
              id={candidate.findingId}
              title={candidate.firstProposal.document.title}
              checked={candidate.findingId === value}
              onChange={onChange}
            />
          ))}
          {offerTyped ? (
            <Candidate
              name={name}
              id={typed}
              title="Use this ID"
              checked={false}
              onChange={onChange}
            />
          ) : null}
        </div>
      ) : null}
      {candidates.isPending ? (
        <p className="decisions-duplicate-note">Loading possible issues…</p>
      ) : candidates.isError ? (
        <p className="decisions-duplicate-note" role="alert">
          The possible issues of this check could not be loaded. You can still
          enter the exact ID of the original.{" "}
          <button
            type="button"
            className="decisions-text-button"
            onClick={() => void candidates.refetch()}
          >
            Try again
          </button>
        </p>
      ) : others.length === 0 ? (
        <p className="decisions-duplicate-note">
          No other possible issue in this check yet. Enter the exact ID of the
          original.
        </p>
      ) : visible.length === 0 && !offerTyped ? (
        <p className="decisions-duplicate-note">
          No possible issue matches. Search by title, or enter an exact ID.
        </p>
      ) : truncated ? (
        <p className="decisions-duplicate-note">
          Showing the first {others.length} possible issues. For another one,
          enter its exact ID.
        </p>
      ) : null}
    </fieldset>
  );
}
