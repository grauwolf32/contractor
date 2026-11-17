import type { ReactNode } from "react";

import { Icon } from "../../app/icon";

/** Title row of a Library section: h2, one line about it and its controls. */
export function LibrarySectionHeader({
  id,
  title,
  description,
  actions,
}: {
  /** Id of the h2, for the section's aria-labelledby. */
  id: string;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="library-section-header">
      <div className="library-section-heading">
        <h2 id={id} className="library-section-title">
          {title}
        </h2>
        {description === undefined ? null : (
          <p className="library-section-lede">{description}</p>
        )}
      </div>
      {actions === undefined ? null : (
        <div className="library-section-actions">{actions}</div>
      )}
    </header>
  );
}

/** A search field named by `label`; the placeholder is the visible hint. */
export function LibrarySearch({
  label,
  placeholder,
  value,
  onChange,
}: {
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="library-search">
      <span className="ui-visually-hidden">{label}</span>
      <Icon name="search" />
      <input
        type="search"
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}
