import type { ReactNode } from "react";
import { Link, useNavigate } from "react-router";

import { Icon, type IconName } from "../../app/icon";
import {
  Button,
  IdChip,
  ListRow,
  ListSection,
  PaneLayout,
  useListNavigation,
} from "../../ui";
import { useLibraryBrowser } from "./browser-state";

export function LibraryBrowser({
  base,
  title,
  list,
  detail,
  selected,
}: {
  base: string;
  title: string;
  list: ReactNode;
  detail: ReactNode;
  selected: boolean;
}) {
  const { listPath, backState, context } = useLibraryBrowser(base);
  return (
    <PaneLayout
      className="library-browser"
      listLabel={title}
      detailLabel={`${title} details`}
      showDetail={selected}
      backLink={{
        to: listPath,
        label: `All ${title.toLowerCase()}`,
        state: backState,
      }}
      list={list}
      detail={
        <>
          {context ? (
            <Link
              className="library-context-back"
              to={context.returnTo}
              state={context.returnState}
            >
              ← {context.returnLabel}
            </Link>
          ) : null}
          {detail}
        </>
      }
    />
  );
}

interface LibraryInventoryItem {
  key: string;
  to: string;
  title: string;
  description: string;
  kind: string;
  icon: IconName;
  tone?: "blue" | "amber" | "green";
  meta: ReactNode;
}

export function LibraryInventory({
  items,
  selected,
  state,
}: {
  items: LibraryInventoryItem[];
  selected: string | undefined;
  state: unknown;
}) {
  const navigate = useNavigate();
  const { containerProps } = useListNavigation({
    count: items.length,
    index: items.findIndex((item) => item.key === selected),
    onMove: (index) => {
      const item = items[index];
      if (item) void navigate(item.to, { state });
    },
  });
  return (
    <div {...containerProps} className="library-inventory-rows">
      <ListSection
        aria-label="Published versions"
        aria-keyshortcuts="J K ArrowDown ArrowUp Enter"
      >
        {items.map((item) => (
          <ListRow
            key={item.key}
            to={item.to}
            state={state}
            selected={item.key === selected}
            clamp={false}
            glyph={
              <span
                className="library-kind-icon"
                data-tone={item.tone ?? "blue"}
              >
                <Icon name={item.icon} />
              </span>
            }
            title={
              <span className="library-inventory-title">
                <span className="library-eyebrow">{item.kind}</span>{" "}
                <strong>{item.title}</strong>{" "}
                <span className="library-inventory-description">
                  {item.description}
                </span>
              </span>
            }
            meta={item.meta}
            trailing={
              <span aria-hidden="true" className="library-inventory-chevron">
                ›
              </span>
            }
          />
        ))}
      </ListSection>
    </div>
  );
}

export function LibraryTag({
  children,
  tone,
}: {
  children: ReactNode;
  tone?: "blue" | "amber" | "green" | undefined;
}) {
  return (
    <span className="library-pill" data-tone={tone}>
      {children}
    </span>
  );
}

export function LibraryFilters({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  options: ReadonlyArray<readonly [string, string]>;
  label: string;
}) {
  return (
    <div className="library-browser-filters" role="group" aria-label={label}>
      {options.map(([id, text]) => (
        <Button
          size="sm"
          key={id}
          aria-pressed={value === id}
          onClick={() => onChange(id)}
        >
          {text}
        </Button>
      ))}
    </div>
  );
}

export function LibraryDetailHeading({
  title,
  kind,
  description,
  selector,
  identityLabel,
  icon,
  tone,
  version,
  meta,
  actions,
}: {
  title: string;
  kind: string;
  description?: string | undefined;
  selector: string;
  identityLabel: string;
  icon: IconName;
  tone?: "blue" | "amber" | "green";
  version: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="library-detail-heading">
      <div className="library-detail-topline">
        <span className="library-eyebrow">{kind}</span>
        <div className="library-detail-version">
          {version}
          {actions}
        </div>
      </div>
      <div className="library-detail-title-line">
        <span
          className="library-kind-icon library-hero-icon"
          data-tone={tone ?? "blue"}
        >
          <Icon name={icon} />
        </span>
        <div>
          <h2>{title}</h2>
          <div className="library-detail-meta">
            {meta}
            <IdChip value={selector} display={selector} label={identityLabel} />
          </div>
        </div>
      </div>
      {description ? (
        <p className="library-detail-description">{description}</p>
      ) : null}
    </header>
  );
}
