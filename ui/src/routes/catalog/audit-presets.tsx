import { useId, useState } from "react";

import { auditPresetPath } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { ContextLink } from "../../app/context-navigation";
import { QueryView } from "../../app/query-view";
import {
  capitalize,
  checkItemKind,
  itemNoun,
  TERMS,
} from "../../app/vocabulary";
import { EmptyState, IdChip } from "../../ui";
import { auditPresetLabel } from "../projects/audits/labels";
import {
  auditModeLabels,
  checkTypeSearchText,
  groupCheckTypes,
  hasFixedItems,
  presetScope,
  useAuditPresets,
  type CheckTypeFamily,
} from "./audit-preset-data";
import {
  CheckTypeAvailability,
  CompatibilityReasons,
} from "./check-type-status";
import { LibrarySearch, LibrarySectionHeader } from "./library-parts";
import { useCatalogQueryState } from "./query-state";

/** What the list hands a check type page as its way back. */
const RETURN_LABEL = "All check types";

/** "View requirements", or "View details" when the items come from inputs. */
function viewLabel(profile: AuditProfile): string {
  return hasFixedItems(profile)
    ? `View ${itemNoun(checkItemKind(profile), 2)}`
    : "View details";
}

function CheckTypeCard({
  family,
  profile,
  onVersion,
}: {
  family: CheckTypeFamily;
  profile: AuditProfile;
  onVersion: (version: string) => void;
}) {
  const heading = useId();
  const { name } = family;
  const selector = `${name}@${profile.ref.version}`;
  const path = auditPresetPath(name, profile.ref.version);
  return (
    <article className="library-card" aria-labelledby={heading}>
      <div className="library-card-top">
        <span className="library-card-kind">
          {auditModeLabels[profile.mode]}
        </span>
        <CheckTypeAvailability profile={profile} size="sm" />
      </div>
      <h3 id={heading} className="library-card-title">
        <ContextLink to={path} returnLabel={RETURN_LABEL}>
          {auditPresetLabel(name)}
        </ContextLink>
      </h3>
      <p className="library-card-text">{presetScope(profile)}</p>
      {profile.serverCompatible ? null : (
        <CompatibilityReasons reasons={profile.compatibilityReasons} />
      )}
      <div className="library-card-meta">
        <label className="library-version">
          <span>Version</span>
          <select
            aria-label={`Version of ${name}`}
            value={profile.ref.version}
            onChange={(event) => onVersion(event.target.value)}
          >
            {family.versions.map((item) => (
              <option key={item.ref.version} value={item.ref.version}>
                {item.ref.version}
              </option>
            ))}
          </select>
        </label>
        <IdChip
          value={selector}
          display={selector}
          label="check type version"
        />
      </div>
      <footer className="library-card-footer">
        <ContextLink
          className="library-card-link"
          to={path}
          returnLabel={RETURN_LABEL}
        >
          {viewLabel(profile)} <span aria-hidden="true">→</span>
        </ContextLink>
      </footer>
    </article>
  );
}

/** Library → Check types: every published check type and its versions. */
export function AuditPresetListRoute() {
  const heading = useId();
  const query = useAuditPresets();
  const state = useCatalogQueryState();
  const [choices, setChoices] = useState<Record<string, string>>({});
  const search = state.committedSearch.toLowerCase();
  const families = groupCheckTypes(query.data ?? []).flatMap((family) => {
    const versions = family.versions.filter((profile) =>
      checkTypeSearchText(profile).includes(search),
    );
    return versions.length === 0 ? [] : [{ ...family, versions }];
  });
  const shown = families.map((family) => ({
    family,
    profile:
      family.versions.find(
        (item) => item.ref.version === choices[family.name],
      ) ?? family.versions[0]!,
  }));
  const unavailable = shown.filter(
    ({ profile }) => !profile.serverCompatible,
  ).length;

  return (
    <section className="library-section" aria-labelledby={heading}>
      <LibrarySectionHeader
        id={heading}
        title={capitalize(TERMS.checkTypes)}
        description="What each type of check works through, and whether this server can run it."
        actions={
          <LibrarySearch
            label="Search check types"
            placeholder="Name, version, standard or mode"
            value={state.draftSearch}
            onChange={state.changeDraftSearch}
          />
        }
      />
      <QueryView
        query={query}
        loading={<p role="status">Loading check types…</p>}
        onRetry={() => void query.refetch()}
      >
        {() =>
          shown.length === 0 ? (
            <div className="library-empty">
              <EmptyState
                title={
                  search
                    ? "No check types match this search."
                    : "No check types published."
                }
              >
                {search
                  ? "Try a name, a version, a standard or a mode."
                  : "Check types appear here once the server publishes them."}
              </EmptyState>
            </div>
          ) : (
            <>
              <p className="library-count" aria-live="polite">
                {shown.length}{" "}
                {shown.length === 1 ? TERMS.checkType : TERMS.checkTypes}
                {unavailable === 0
                  ? null
                  : ` · ${unavailable} unavailable on this server`}
              </p>
              <div className="library-grid">
                {shown.map(({ family, profile }) => (
                  <CheckTypeCard
                    key={family.name}
                    family={family}
                    profile={profile}
                    onVersion={(version) =>
                      setChoices((previous) => ({
                        ...previous,
                        [family.name]: version,
                      }))
                    }
                  />
                ))}
              </div>
            </>
          )
        }
      </QueryView>
    </section>
  );
}
