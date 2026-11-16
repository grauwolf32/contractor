import { useLibraryBrowser } from "./browser-state";
import { useId } from "react";
import { useParams, useSearchParams } from "react-router";
import { auditPresetPath } from "../../api/audit-presets";
import { ENGAGEMENT_MEDIA_TYPE } from "../../api/artifacts";
import { plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { EmptyState } from "../../ui";
import { presentCheckType, inputLabel } from "../checks/start/check-types";
import {
  checkTypeSearchText,
  groupCheckTypes,
  useAuditPresets,
} from "./audit-preset-data";
import { AuditPresetDetailRoute } from "./audit-preset-detail";
import {
  LibraryBrowser,
  LibraryFilters,
  LibraryInventory,
  LibraryTag,
} from "./browser";
import { LibrarySearch, LibrarySectionHeader } from "./library-parts";
import { useCatalogQueryState } from "./query-state";

const BASE = "/catalog/audit-presets";
export function AuditPresetListRoute() {
  const { name, version } = useParams();
  const heading = useId();
  const query = useAuditPresets();
  const state = useCatalogQueryState(BASE);
  const browser = useLibraryBrowser(BASE);
  const [params, setParams] = useSearchParams();
  const mode = params.get("mode") ?? "all";
  const material = params.get("material") ?? "all";
  const families = groupCheckTypes(query.data ?? []);
  const materials = [
    ...new Set((query.data ?? []).flatMap((p) => Object.keys(p.inputs))),
  ].sort();
  const shown = families
    .flatMap((family) => {
      const versions = family.versions.filter((profile) => {
        const http = Object.values(profile.inputs).some((i) =>
          i.media_types.includes(ENGAGEMENT_MEDIA_TYPE),
        );
        return (
          checkTypeSearchText(profile).includes(
            state.committedSearch.toLowerCase(),
          ) &&
          (mode === "all" ||
            (mode === "http"
              ? http
              : Object.hasOwn(profile.inputs, "source"))) &&
          (material === "all" || Object.hasOwn(profile.inputs, material))
        );
      });
      const profile =
        versions.find(
          (p) => p.ref.version === (name === family.name ? version : undefined),
        ) ?? versions[0];
      return profile ? [{ family, profile }] : [];
    })
    .sort(
      (a, b) =>
        presentCheckType(a.profile).order - presentCheckType(b.profile).order ||
        a.family.name.localeCompare(b.family.name),
    );
  const first = shown[0]?.profile;
  const selectedName = name ?? first?.ref.name;
  const selectedVersion =
    name !== undefined ? (version ?? "") : first?.ref.version;
  function filter(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value === "all") next.delete(key);
    else next.set(key, value);
    setParams(next, { replace: true, state: browser.location.state });
  }
  return (
    <LibraryBrowser
      base={BASE}
      title="Check types"
      selected={name !== undefined}
      list={
        <div className="library-inventory">
          <LibrarySectionHeader
            id={heading}
            title="Find your next check"
            description="Choose by goal and materials."
            actions={<span className="library-count-pill">{shown.length}</span>}
          />
          <LibrarySearch
            label="Search check types"
            placeholder="Search checks…"
            value={state.draftSearch}
            onChange={state.changeDraftSearch}
          />
          <LibraryFilters
            label="Check type filters"
            value={mode}
            onChange={(v) => filter("mode", v)}
            options={[
              ["all", "All types"],
              ["source", "Source"],
              ["http", "Engagement"],
            ]}
          />
          <label className="library-material-filter">
            Material
            <select
              className="ui-control"
              data-size="xs"
              aria-label="Material"
              value={material}
              onChange={(e) => filter("material", e.target.value)}
            >
              <option value="all">Any material</option>
              {materials.map((m) => (
                <option key={m} value={m}>
                  {inputLabel(m)}
                </option>
              ))}
            </select>
          </label>
          <QueryView
            query={query}
            loading={<p role="status">Loading check types…</p>}
          >
            {() =>
              shown.length === 0 ? (
                <EmptyState title="No matching check types">
                  Try another name or material.
                </EmptyState>
              ) : (
                <LibraryInventory
                  selected={selectedName}
                  state={browser.rowState}
                  items={shown.map(({ family, profile }) => {
                    const p = presentCheckType(profile);
                    const http = Object.values(profile.inputs).some((i) =>
                      i.media_types.includes(ENGAGEMENT_MEDIA_TYPE),
                    );
                    return {
                      key: family.name,
                      to:
                        auditPresetPath(family.name, profile.ref.version) +
                        browser.search,
                      title: p.label,
                      description: p.description,
                      kind: "Check type",
                      icon: p.icon,
                      tone: http ? "amber" : "blue",
                      meta: (
                        <>
                          <LibraryTag tone={http ? "amber" : "blue"}>
                            {http ? "Engagement" : "Materials"}
                          </LibraryTag>
                          <span>
                            {plural(
                              Object.keys(profile.inputs).length,
                              "input",
                            )}
                          </span>
                        </>
                      ),
                    };
                  })}
                />
              )
            }
          </QueryView>
        </div>
      }
      detail={
        selectedName !== undefined && selectedVersion !== undefined ? (
          <AuditPresetDetailRoute
            name={selectedName}
            version={selectedVersion}
          />
        ) : (
          <EmptyState title="Choose a check type">
            Its purpose and materials will appear here.
          </EmptyState>
        )
      }
    />
  );
}
