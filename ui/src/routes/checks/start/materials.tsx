import { useId } from "react";
import { Link } from "react-router";

import type { ArtifactMetadata } from "../../../api/artifacts";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { formatName, inSentence, materialKey } from "./check-types";
import { CheckIcon } from "./icons";
import { projectPaths } from "./paths";
import type { InputSlot, StartCheck } from "./use-start-check";

/** Labels of inputs that share one material, per shared material. */
function sharedMaterials(slots: readonly InputSlot[]): string[][] {
  const byMaterial = new Map<string, string[]>();
  for (const slot of slots) {
    if (slot.selected === undefined) continue;
    const key = materialKey(slot.selected);
    byMaterial.set(key, [...(byMaterial.get(key) ?? []), slot.label]);
  }
  return [...byMaterial.values()].filter((labels) => labels.length > 1);
}

function joinWords(words: readonly string[]): string {
  return words.length <= 1
    ? (words[0] ?? "")
    : `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}

function exactName(material: ArtifactMetadata): string {
  const { namespace, name, revision } = material.artifact;
  return `${namespace}/${name}@${revision}`;
}

/** The exact material: namespace/name@revision, its format and size. */
function MaterialRef({ material }: { material: ArtifactMetadata }) {
  return (
    <span className="start-material-ref">
      <code>{exactName(material)}</code>
      <span>
        {formatName(material.mediaType)}, {formatBytes(material.size)}
      </span>
    </span>
  );
}

function MaterialSlot({
  slot,
  complete,
  addHref,
  onChoose,
}: {
  slot: InputSlot;
  complete: boolean;
  addHref: string;
  onChoose: (input: string, value: string) => void;
}) {
  const { candidates, selected, input } = slot;
  // A single format match shows as attached; anything else is a choice.
  const choosing =
    candidates.length > 1 ||
    (candidates.length === 1 && selected === undefined);
  return (
    <li
      className="start-material"
      data-state={selected === undefined ? "open" : "attached"}
    >
      <span className="start-material-mark" aria-hidden="true">
        {selected === undefined ? null : <CheckIcon />}
      </span>
      <div className="start-material-main">
        <div className="start-material-head">
          <span className="start-material-label">{slot.label}</span>
          <span className="start-quiet">
            {slot.formats}
            {input.required ? "" : ", optional"}
          </span>
        </div>
        {selected !== undefined && !choosing ? (
          <MaterialRef material={selected} />
        ) : null}
        {choosing ? (
          <select
            className="start-input start-select"
            aria-label={`Material for ${slot.label}`}
            value={slot.selection}
            onChange={(event) => onChoose(slot.name, event.target.value)}
          >
            <option value="">
              {input.required ? "Choose a material" : "Not used"}
            </option>
            {candidates.map((candidate) => (
              <option
                key={materialKey(candidate)}
                value={materialKey(candidate)}
              >
                {exactName(candidate)} · {candidate.mediaType}
              </option>
            ))}
          </select>
        ) : null}
        {candidates.length === 0 && complete ? (
          <p className="start-quiet start-tight">
            {input.required
              ? "No material in this format yet. "
              : "Not used: no material in this format. "}
            {input.required ? <Link to={addHref}>Add one</Link> : null}
          </p>
        ) : null}
        {selected === undefined ? null : (
          <p className="start-quiet start-tight">
            {slot.automatic
              ? "Attached: the only material in a matching format."
              : "Format matches."}
          </p>
        )}
      </div>
      {!input.required && selected !== undefined && !choosing ? (
        <button
          type="button"
          className="ui-btn"
          data-variant="ghost"
          data-size="xs"
          aria-label={`Don't use ${slot.label}`}
          onClick={() => onChoose(slot.name, "")}
        >
          Don&apos;t use
        </button>
      ) : null}
    </li>
  );
}

function LiveTargetMaterial({
  url,
  settings,
}: {
  url: string | undefined;
  settings: string;
}) {
  return (
    <li
      className="start-material"
      data-state={url === undefined ? "open" : "attached"}
    >
      <span className="start-material-mark" aria-hidden="true">
        {url === undefined ? null : <CheckIcon />}
      </span>
      <div className="start-material-main">
        <div className="start-material-head">
          <span className="start-material-label">Live target</span>
          <span className="start-quiet">project setting</span>
        </div>
        {url === undefined ? (
          <p className="start-quiet start-tight">
            Not set for this project.{" "}
            <Link to={settings}>Set the live target</Link> or enter a target for
            this check below.
          </p>
        ) : (
          <span className="start-material-ref">
            <code>{url}</code>
          </span>
        )}
      </div>
    </li>
  );
}

/** Inputs attached from the project's current materials. */
export function MaterialsCard({ model }: { model: StartCheck }) {
  const headingId = useId();
  const { materials, slots, exact } = model;
  const paths = projectPaths(model.projectId);
  const usesTarget = model.scopeFields.has("target");
  const shared = sharedMaterials(slots);
  return (
    <section className="start-card" aria-labelledby={headingId}>
      <div className="start-card-head">
        <h3 id={headingId}>Materials</h3>
        <span className="start-quiet">Attached from the project</span>
      </div>
      {materials.loading ? (
        <p className="start-quiet start-tight" role="status">
          Loading the project&apos;s materials…
        </p>
      ) : null}
      {materials.error === null ? null : (
        <div className="start-stack">
          <ErrorNotice
            error={materials.error}
            context={
              materials.items === undefined
                ? "The project's materials could not be loaded."
                : materials.moreFailed
                  ? "More of the project's materials could not be loaded."
                  : "The project's materials could not be refreshed. The page uses the ones read before."
            }
          />
          {/* Without any page, the list pane offers the retry. */}
          {materials.items === undefined ? null : (
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              disabled={materials.retrying}
              onClick={materials.retry}
            >
              Retry loading materials
            </button>
          )}
        </div>
      )}
      {slots.length === 0 && !usesTarget ? null : (
        <ul role="list" className="start-material-list">
          {slots.map((slot) => (
            <MaterialSlot
              key={slot.name}
              slot={slot}
              complete={materials.complete}
              addHref={paths.addMaterial}
              onChoose={model.chooseMaterial}
            />
          ))}
          {usesTarget ? (
            <LiveTargetMaterial
              url={model.project?.httpTarget?.url}
              settings={paths.settings}
            />
          ) : null}
        </ul>
      )}
      {shared.map((labels) => (
        <p
          key={labels.join("\n")}
          className="notice notice-warning start-tight"
        >
          The same material is attached to {joinWords(labels.map(inSentence))}.
          Its format fits each of them; check that it is right for all of them.
        </p>
      ))}
      {materials.hasMore ? (
        <div className="start-stack">
          <p className="start-quiet start-tight">
            Showing the first {materials.items?.length ?? 0} materials. Load
            more to choose from the rest.
          </p>
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            disabled={materials.loadingMore}
            onClick={materials.loadMore}
          >
            {materials.loadingMore ? "Loading…" : "Load more materials"}
          </button>
        </div>
      ) : null}
      <p className="start-card-note">
        Picked by file format: a format match does not check what is inside, so
        make sure each one is right.
        {exact?.requiresInputValidation === true
          ? " The server checks them when the check starts."
          : ""}{" "}
        <Link to={paths.addMaterial}>Add materials</Link>
      </p>
    </section>
  );
}
