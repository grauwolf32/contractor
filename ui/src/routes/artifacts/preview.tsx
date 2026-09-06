import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useId,
  useState,
  type ReactNode,
} from "react";

import { canPreviewArtifact, type ArtifactMetadata } from "../../api/artifacts";
import { ErrorNotice } from "./common";
import {
  createArtifactPreviewPlan,
  type ArtifactPreviewPlan,
} from "./preview-plan";

const MarkdownArtifactPreview = lazy(() => import("./previews/markdown"));
const OpenApiArtifactPreview = lazy(() => import("./previews/openapi"));
const LikeC4ArtifactPreview = lazy(() => import("./previews/likec4"));
const DiffArtifactPreview = lazy(() => import("./previews/diff"));

const RENDERER_LABELS = {
  diff: "Diff",
  markdown: "Markdown",
  openapi: "OpenAPI",
  likec4: "LikeC4",
} as const;

type RenderedPlan = Exclude<ArtifactPreviewPlan, { kind: "source" }>;

function rendererLabel(plan: RenderedPlan): string {
  return RENDERER_LABELS[plan.kind];
}

function Renderer({ plan, source }: { plan: RenderedPlan; source: string }) {
  switch (plan.kind) {
    case "diff":
      return <DiffArtifactPreview source={source} />;
    case "markdown":
      return <MarkdownArtifactPreview source={source} />;
    case "openapi":
      return <OpenApiArtifactPreview document={plan.document} />;
    case "likec4":
      return <LikeC4ArtifactPreview source={source} />;
  }
}

class RendererErrorBoundary extends Component<
  { children: ReactNode },
  { message?: string }
> {
  override state: { message?: string } = {};

  static getDerivedStateFromError(error: unknown): { message: string } {
    return {
      message:
        error instanceof Error ? error.message : "Artifact rendering failed",
    };
  }

  override render(): ReactNode {
    if (this.state.message !== undefined) {
      return (
        <p className="artifact-renderer-error" role="alert">
          {this.state.message}
        </p>
      );
    }
    return this.props.children;
  }
}

function LoadedArtifactPreview({
  mediaType,
  source,
}: {
  mediaType: string;
  source: string;
}) {
  const [plan, setPlan] = useState<ArtifactPreviewPlan>();
  const [tab, setTab] = useState<"rendered" | "source">("rendered");
  const id = useId();

  useEffect(() => {
    let active = true;
    void createArtifactPreviewPlan(mediaType, source)
      .then((nextPlan) => {
        if (active) {
          setPlan(nextPlan);
        }
      })
      .catch(() => {
        if (active) {
          setPlan({ kind: "source" });
        }
      });
    return () => {
      active = false;
    };
  }, [mediaType, source]);

  if (plan === undefined) {
    return <p className="artifact-renderer-loading">Preparing preview…</p>;
  }
  if (plan.kind === "source") {
    return (
      <pre className="artifact-preview" tabIndex={0}>
        {source}
      </pre>
    );
  }

  const renderedTabId = `${id}-rendered-tab`;
  const renderedPanelId = `${id}-rendered-panel`;
  const sourceTabId = `${id}-source-tab`;
  const sourcePanelId = `${id}-source-panel`;
  return (
    <div className="artifact-preview-shell">
      <div
        aria-label="Artifact preview mode"
        className="artifact-preview-tabs"
        role="tablist"
      >
        <button
          aria-controls={renderedPanelId}
          aria-selected={tab === "rendered"}
          className={tab === "rendered" ? "selected" : undefined}
          id={renderedTabId}
          role="tab"
          type="button"
          onClick={() => setTab("rendered")}
        >
          Rendered
        </button>
        <button
          aria-controls={sourcePanelId}
          aria-selected={tab === "source"}
          className={tab === "source" ? "selected" : undefined}
          id={sourceTabId}
          role="tab"
          type="button"
          onClick={() => setTab("source")}
        >
          Source
        </button>
        <span>{rendererLabel(plan)}</span>
      </div>
      <div
        aria-labelledby={renderedTabId}
        hidden={tab !== "rendered"}
        id={renderedPanelId}
        role="tabpanel"
      >
        <RendererErrorBoundary>
          <Suspense
            fallback={
              <p className="artifact-renderer-loading">Loading renderer…</p>
            }
          >
            <Renderer plan={plan} source={source} />
          </Suspense>
        </RendererErrorBoundary>
      </div>
      <div
        aria-labelledby={sourceTabId}
        hidden={tab !== "source"}
        id={sourcePanelId}
        role="tabpanel"
      >
        <pre className="artifact-preview" tabIndex={0}>
          {source}
        </pre>
      </div>
    </div>
  );
}

export function ArtifactPreviewPanel({
  metadata,
  loadPreview,
  loadOnMountKey,
  unavailableCopy,
}: {
  metadata: ArtifactMetadata;
  loadPreview: () => Promise<string>;
  loadOnMountKey?: readonly unknown[];
  unavailableCopy: string;
}) {
  const manualPreview = useMutation({ mutationFn: loadPreview });
  const canPreview = canPreviewArtifact(metadata);
  const automaticPreview = useQuery({
    queryKey: loadOnMountKey ?? ["artifact-preview", "manual"],
    queryFn: loadPreview,
    enabled: loadOnMountKey !== undefined && canPreview,
  });
  const automatic = loadOnMountKey !== undefined;
  const data = automatic ? automaticPreview.data : manualPreview.data;
  const error = automatic ? automaticPreview.error : manualPreview.error;
  const pending = automatic
    ? automaticPreview.isPending || automaticPreview.isFetching
    : manualPreview.isPending;

  function requestPreview(): void {
    if (automatic) {
      void automaticPreview.refetch();
      return;
    }
    manualPreview.mutate();
  }

  return (
    <div className="panel artifact-preview-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Safe rendering</p>
          <h3>Preview</h3>
        </div>
        {!canPreview && automatic ? null : (
          <button
            className="secondary-button"
            type="button"
            disabled={!canPreview || pending}
            onClick={requestPreview}
          >
            {pending
              ? "Loading…"
              : data === undefined
                ? automatic
                  ? "Retry preview"
                  : "Load preview"
                : "Reload preview"}
          </button>
        )}
      </div>
      {canPreview ? (
        <p className="muted-copy">
          Preview is capped at 256 KiB. Supported documents render locally;
          source remains available.
        </p>
      ) : (
        <div className="compact-empty">{unavailableCopy}</div>
      )}
      {error === null ? null : <ErrorNotice error={error} />}
      {data === undefined ? null : (
        <LoadedArtifactPreview
          key={metadata.artifact.revision}
          mediaType={metadata.mediaType}
          source={data}
        />
      )}
    </div>
  );
}
