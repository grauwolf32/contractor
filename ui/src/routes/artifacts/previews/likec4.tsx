import { LikeC4Model } from "@likec4/core/model";
import type { LikeC4ModelDump } from "@likec4/core/types";
import {
  LikeC4ModelProvider,
  ReactLikeC4,
  type LikeC4DiagramProps,
} from "@likec4/diagram";
import { useEffect, useMemo, useState } from "react";

import type { LikeC4WorkerResponse } from "./likec4-protocol";

type Diagram = LikeC4DiagramProps["view"];
type DiagramModel = ReturnType<typeof LikeC4Model.fromDump>;

interface DiagramEntry {
  key: string;
  model: DiagramModel;
  view: Diagram;
}

type DiagramState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; diagrams: DiagramEntry[] };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "LikeC4 rendering failed";
}

export default function LikeC4ArtifactPreview({ source }: { source: string }) {
  const [state, setState] = useState<DiagramState>({ status: "loading" });
  const [selectedView, setSelectedView] = useState<string>();

  useEffect(() => {
    const worker = new Worker(new URL("./likec4.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<LikeC4WorkerResponse>) => {
      if (!event.data.ok) {
        setState({ status: "error", message: event.data.message });
        return;
      }
      try {
        const models = event.data.models.map((dump) =>
          LikeC4Model.fromDump(dump as LikeC4ModelDump),
        );
        const diagrams = models.flatMap((model) =>
          [...model.views()].map((view) => ({
            key: `${model.projectId}:${view.id}`,
            model,
            view: view.$layouted as Diagram,
          })),
        );
        setState({ status: "ready", diagrams });
        setSelectedView(diagrams[0]?.key);
      } catch (error) {
        setState({ status: "error", message: errorMessage(error) });
      }
    };
    worker.onerror = (event) => {
      setState({
        status: "error",
        message: event.message || "LikeC4 rendering worker failed",
      });
    };
    worker.postMessage(source);
    return () => worker.terminate();
  }, [source]);

  const diagram = useMemo(
    () =>
      state.status === "ready"
        ? state.diagrams.find((candidate) => candidate.key === selectedView)
        : undefined,
    [selectedView, state],
  );

  if (state.status === "loading") {
    return <p className="artifact-renderer-loading">Laying out diagram…</p>;
  }
  if (state.status === "error") {
    return (
      <pre className="artifact-renderer-error" role="alert">
        {state.message}
      </pre>
    );
  }
  if (state.diagrams.length === 0 || diagram === undefined) {
    return <div className="compact-empty">This LikeC4 model has no views.</div>;
  }

  return (
    <div className="likec4-artifact-preview" data-mantine-color-scheme="dark">
      <div className="artifact-renderer-toolbar">
        <label>
          View
          <select
            value={selectedView}
            onChange={(event) => setSelectedView(event.target.value)}
          >
            {state.diagrams.map((candidate) => (
              <option key={candidate.key} value={candidate.key}>
                {candidate.view.title ?? candidate.view.id}
              </option>
            ))}
          </select>
        </label>
        <span>External icons are omitted</span>
      </div>
      <LikeC4ModelProvider likec4model={diagram.model}>
        <ReactLikeC4
          key={diagram.key}
          className="likec4-artifact-canvas"
          viewId={diagram.view.id}
          background="dots"
          colorScheme="dark"
          controls
          enableElementDetails={false}
          enableRelationshipBrowser={false}
          enableRelationshipDetails={false}
          enableSearch={false}
          fitView
          injectFontCss={false}
          keepAspectRatio={false}
          onNavigateTo={(viewId) => {
            const target = String(viewId);
            const candidate = state.diagrams.find(
              (entry) =>
                entry.model === diagram.model && entry.view.id === target,
            );
            if (candidate !== undefined) {
              setSelectedView(candidate.key);
            }
          }}
          pannable
          reduceGraphics="auto"
          renderIcon={() => null}
          showNavigationButtons={false}
          zoomable
        />
      </LikeC4ModelProvider>
    </div>
  );
}
