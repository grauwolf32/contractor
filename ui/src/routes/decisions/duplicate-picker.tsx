import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { listProjectFindings } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { collectAuditPages } from "../../api/audit-collections";
import { Button } from "../../ui";

export function DuplicatePicker({
  projectId,
  findingId,
  value,
  onChange,
  searchId,
}: {
  projectId: string;
  findingId: string;
  value: string;
  onChange: (id: string) => void;
  searchId?: string;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const generatedId = useId();
  const inputId = searchId ?? generatedId;
  const [search, setSearch] = useState("");
  const queryKey = ["project-findings", projectId, "duplicate-targets"];
  const candidates = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      collectAuditPages(
        (cursor) =>
          listProjectFindings(
            api,
            projectId,
            cursor === undefined ? {} : { cursor },
          ),
        pageParam === undefined ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (last) => (last.truncated ? last.nextCursor : undefined),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: false,
  });
  const choices = (
    candidates.data?.pages.flatMap((page) => page.items) ?? []
  ).filter(
    (item) =>
      item.findingId !== findingId &&
      (item.title + " " + item.findingId)
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  return (
    <div className="decisions-duplicate-picker">
      <label htmlFor={inputId}>Original possible issue</label>
      <input
        id={inputId}
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        placeholder="Search this Project"
      />
      <label>
        Finding ID
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      </label>
      {candidates.error === null ? null : (
        <p role="alert">
          Candidates could not be loaded. You can enter an ID, or{" "}
          <Button
            size="xs"
            onClick={() => void client.resetQueries({ queryKey, exact: true })}
          >
            restart the list
          </Button>
          .
        </p>
      )}
      <p className="ui-caption">
        Search covers loaded candidates. You can also enter an exact Finding ID.
      </p>
      <ul>
        {choices.slice(0, 20).map((item) => (
          <li key={item.findingId}>
            <Button size="xs" onClick={() => onChange(item.findingId)}>
              {item.title}
            </Button>
          </li>
        ))}
      </ul>
      {choices.length > 20 ? (
        <p className="ui-caption">Refine the search to see more matches.</p>
      ) : null}
      {candidates.hasNextPage ? (
        <Button
          size="xs"
          pending={candidates.isFetchingNextPage}
          onClick={() => void candidates.fetchNextPage()}
        >
          Load more candidates
        </Button>
      ) : null}
    </div>
  );
}
