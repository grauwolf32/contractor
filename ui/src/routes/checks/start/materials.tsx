import { useId } from "react";
import { Link } from "react-router";

import type { ArtifactMetadata } from "../../../api/artifacts";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { Button } from "../../../ui";
import { formatName, inSentence, joinWords, materialKey } from "./check-types";
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
        {selected === undefined ? null : (
          <Icon name="check" size={14} strokeWidth={2.2} />
        )}
      </span>
      <div className="start-material-main">
        <div className="start-material-head">
          <span className="start-material-label">{slot.label}</span>
          <span className="ui-caption">
            {slot.formats}
            {input.required ? "" : ", optional"}
          </span>
        </div>
        {selected !== undefined && !choosing ? (
          <MaterialRef material={selected} />
        ) : null}
        {choosing ? (
          <select
            className="start-material-select"
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
          <p className="ui-caption">
            {input.required
              ? "No material in this format yet. "
              : "Not used: no material in this format. "}
            {input.required ? <Link to={addHref}>Add one</Link> : null}
          </p>
        ) : null}
        {selected === undefined ? null : (
          <p className="ui-caption">
            {slot.automatic
              ? "Attached: the only material in a matching format."
              : "Format matches."}
          </p>
        )}
      </div>
      {!input.required && selected !== undefined && !choosing ? (
        <Button
          variant="ghost"
          size="xs"
          aria-label={`Don't use ${slot.label}`}
          onClick={() => onChoose(slot.name, "")}
        >
          Don&apos;t use
        </Button>
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
        {url === undefined ? null : (
          <Icon name="check" size={14} strokeWidth={2.2} />
        )}
      </span>
      <div className="start-material-main">
        <div className="start-material-head">
          <span className="start-material-label">Live target</span>
          <span className="ui-caption">project setting</span>
        </div>
        {url === undefined ? (
          <p className="ui-caption">
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
        <span className="ui-caption">Attached from the project</span>
      </div>
      {materials.loading ? (
        <p className="ui-caption" role="status">
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
            <Button
              size="sm"
              pending={materials.retrying}
              onClick={materials.retry}
            >
              Retry loading materials
            </Button>
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
          <p className="ui-caption">
            Showing the first {materials.items?.length ?? 0} materials. Load
            more to choose from the rest.
          </p>
          <Button
            size="sm"
            pending={materials.loadingMore}
            pendingLabel="Loading…"
            onClick={materials.loadMore}
          >
            Load more materials
          </Button>
        </div>
      ) : null}
      <p className="ui-caption">
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
