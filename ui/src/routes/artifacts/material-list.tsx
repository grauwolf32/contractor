import type { ReactNode } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { components } from "../../api/generated/public";
import { CursorControls } from "../../app/cursor-controls";
import type {
  CursorStackControls,
  PageContinuation,
} from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { Button, EmptyState } from "../../ui";
import { useNamespaceFilter } from "./namespace-filter";
import "./materials.css";

interface MaterialPage {
  items: ArtifactMetadata[];
  page: components["schemas"]["PageInfo"];
}

/**
 * The body of a Files or Materials panel: the namespace filter, one page of
 * the list in its loading, error and empty states, and the page cursor.
 * `noun` names the items in the copy ("files", "materials").
 */
export function MaterialList<P extends MaterialPage>({
  query,
  namespace,
  onNamespace,
  noun,
  empty,
  namespaceEmptyText,
  pagesLabel,
  pages,
  children,
}: {
  query: {
    data: P | undefined;
    error: Error | null;
    isFetching: boolean;
    refetch: () => unknown;
  };
  namespace: string | undefined;
  /** Applies a namespace filter, or undefined for every namespace. */
  onNamespace: (namespace: string | undefined) => void;
  noun: "files" | "materials";
  /** The empty list without a namespace filter. */
  empty: { title: string; text: ReactNode };
  namespaceEmptyText: string;
  pagesLabel: string;
  pages: {
    controls: (page: PageContinuation | undefined) => CursorStackControls;
  };
  children: (page: P) => ReactNode;
}) {
  const filter = useNamespaceFilter({ value: namespace, onApply: onNamespace });
  return (
    <>
      <div className="materials-panel-head">
        {filter.form}
        {namespace === undefined ? null : (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onNamespace(undefined)}
          >
            Show all namespaces
          </Button>
        )}
      </div>
      {filter.error}
      <QueryView
        query={query}
        loading={
          <p className="materials-loading" role="status">
            {`Loading ${noun}…`}
          </p>
        }
        errorContext={`Could not load ${noun}`}
        isEmpty={(page) => page.items.length === 0}
        empty={
          namespace === undefined ? (
            <EmptyState title={empty.title}>{empty.text}</EmptyState>
          ) : (
            <EmptyState title={`Nothing in the ${namespace} namespace`}>
              {namespaceEmptyText}
            </EmptyState>
          )
        }
      >
        {children}
      </QueryView>
      <div className="materials-pager">
        <CursorControls
          label={pagesLabel}
          {...pages.controls(query.data?.page)}
        />
      </div>
    </>
  );
}
