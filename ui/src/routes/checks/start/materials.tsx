import { materialKey } from "../../artifacts/material-library-filter";
import { AssertionMaterialEditor } from "./assertion-material-editor";
import { useId, useState, type ReactNode } from "react";
import { Link } from "react-router";

import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { Button, IconButton } from "../../../ui";
import { AddMaterialSheet } from "../../artifacts/add-material";
import { MaterialKindMark } from "../../artifacts/icons";
import { materialKindOf } from "../../artifacts/kinds";
import { MaterialSummary } from "../../artifacts/material-chooser";
import { MaterialPicker } from "./material-picker";
import { inSentence, joinWords } from "./check-types";
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

function MaterialSlot({
  slot,
  complete,
  onChoose,
  onPreview,
  onClear,
  children,
}: {
  slot: InputSlot;
  complete: boolean;
  onChoose: () => void;
  onPreview: () => void;
  onClear: () => void;
  children?: ReactNode;
}) {
  const { selected, input } = slot;
  return (
    <li
      className="start-material check-material-slot"
      data-state={selected === undefined ? "open" : "attached"}
    >
      <div className="start-material-head">
        <span className="start-material-label">{slot.label}</span>
        <span className="ui-caption">
          {input.required ? "Required" : "Optional"}
        </span>
      </div>
      <div className="check-material-attachment">
        {selected === undefined ? (
          <span className="ui-caption">
            {complete && slot.candidates.length === 0
              ? "No material in this format yet."
              : "Choose a material"}
          </span>
        ) : (
          <>
            <MaterialKindMark kind={materialKindOf(selected)} />
            <MaterialSummary material={selected} />
          </>
        )}
        <div className="ui-toolbar check-material-actions">
          {selected === undefined ? null : (
            <IconButton
              size="sm"
              variant="ghost"
              label={`Preview ${inSentence(slot.label)}`}
              icon={<Icon name="docs" />}
              onClick={onPreview}
            />
          )}
          <Button
            size="sm"
            variant={selected === undefined ? "secondary" : "ghost"}
            aria-label={`${selected === undefined ? "Choose" : "Change"} ${inSentence(slot.label)} material`}
            onClick={onChoose}
          >
            {selected === undefined ? "Choose" : "Change"}
          </Button>
          {!input.required && selected !== undefined ? (
            <IconButton
              size="xs"
              variant="ghost"
              label={`Don't use ${slot.label}`}
              icon={<Icon name="close" size={14} />}
              onClick={onClear}
            />
          ) : null}
        </div>
      </div>
      {selected === undefined ? (
        <p className="ui-caption">Accepted formats: {slot.formats}.</p>
      ) : (
        <p className="ui-visually-hidden">
          {slot.automatic
            ? "Attached: the only material in a matching format."
            : "Format matches."}
        </p>
      )}
      {children}
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
  const [picker, setPicker] = useState<{
    input: string;
    preview: boolean;
    context: string;
  }>();
  const [adding, setAdding] = useState(false);
  const context = `${model.projectId}:${model.selection?.profile.ref.digest ?? ""}`;
  const activeSlot =
    picker?.context === context
      ? model.slots.find((slot) => slot.name === picker.input)
      : undefined;
  const { materials, slots } = model;
  const paths = projectPaths(model.projectId);
  const usesTarget = model.scopeFields.has("target");
  const shared = sharedMaterials(slots);
  return (
    <section
      className="start-card start-materials-card"
      aria-labelledby={headingId}
    >
      <div className="start-card-head">
        <h3 id={headingId}>Materials</h3>
        <Button
          size="sm"
          variant="ghost"
          icon={<Icon name="plus" />}
          onClick={() => setAdding(true)}
        >
          Add
        </Button>
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
              onChoose={() =>
                setPicker({ input: slot.name, preview: false, context })
              }
              onPreview={() =>
                setPicker({ input: slot.name, preview: true, context })
              }
              onClear={() => model.chooseMaterial(slot.name, "")}
            >
              {model.exact?.program?.seed_source?.input === slot.name ? (
                <AssertionMaterialEditor
                  key={`${model.projectId}:${model.exact.ref.digest}`}
                  model={model}
                  input={slot.name}
                />
              ) : null}
            </MaterialSlot>
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
        Choose the saved version to use for this check. You can preview each
        material before attaching it.
      </p>
      {activeSlot === undefined || picker === undefined ? null : (
        <MaterialPicker
          key={`${context}:${activeSlot.name}`}
          model={model}
          slot={activeSlot}
          previewInitially={picker.preview}
          onClose={() => setPicker(undefined)}
        />
      )}
      {adding ? (
        <AddMaterialSheet
          projectId={model.projectId}
          onClose={() => setAdding(false)}
          onAdded={() => setAdding(false)}
        />
      ) : null}
    </section>
  );
}
