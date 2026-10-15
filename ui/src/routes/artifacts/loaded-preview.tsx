import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import {
  createArtifactPreviewPlan,
  type ArtifactPreviewPlan,
} from "./preview-plan";
import { ScanReportSummary } from "./scan-report-summary";
import type { HTTPPreviewContext } from "./previews/http-exchange";
import { PanelTabs } from "./tabs";
import "./materials.css";

const MarkdownArtifactPreview = lazy(() => import("./previews/markdown"));
const OpenApiArtifactPreview = lazy(() => import("./previews/openapi"));
const LikeC4ArtifactPreview = lazy(() => import("./previews/likec4"));
const DiffArtifactPreview = lazy(() => import("./previews/diff"));
const HTTPExchangePreview = lazy(() => import("./previews/http-exchange"));

const RENDERER_LABELS = {
  diff: "Diff",
  markdown: "Markdown",
  openapi: "OpenAPI",
  likec4: "LikeC4",
  http: "HTTP exchange",
  "invalid-http": "HTTP exchange",
} as const;

type RenderedPlan = Exclude<ArtifactPreviewPlan, { kind: "source" }>;

function Renderer({
  plan,
  source,
  context,
}: {
  plan: RenderedPlan;
  source: string;
  context?: HTTPPreviewContext | undefined;
}) {
  switch (plan.kind) {
    case "http":
      return <HTTPExchangePreview document={plan.document} context={context} />;
    case "invalid-http":
      return (
        <p className="artifact-renderer-error" role="alert">
          This HTTP exchange is invalid or unsupported. Download the saved file
          to inspect its bytes.
        </p>
      );
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
        error instanceof Error
          ? error.message
          : "The preview could not be rendered",
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

export function LoadedArtifactPreview({
  mediaType,
  source,
  context,
}: {
  mediaType: string;
  source: string;
  context?: HTTPPreviewContext | undefined;
}) {
  const [plan, setPlan] = useState<ArtifactPreviewPlan>();

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
      <div className="materials-preview-source">
        {mediaType === "application/json" ? (
          <ScanReportSummary source={source} />
        ) : null}
        <pre className="artifact-preview" tabIndex={0}>
          {source}
        </pre>
      </div>
    );
  }

  return (
    <div className="materials-preview-shell">
      <PanelTabs
        label="Preview mode"
        note={RENDERER_LABELS[plan.kind]}
        tabs={[
          {
            id: "rendered",
            label: "Rendered",
            panel: (
              <RendererErrorBoundary>
                <Suspense
                  fallback={
                    <p className="artifact-renderer-loading">
                      Loading renderer…
                    </p>
                  }
                >
                  <Renderer plan={plan} source={source} context={context} />
                </Suspense>
              </RendererErrorBoundary>
            ),
          },
          {
            id: "source",
            label: "Source",
            panel: (
              <pre className="artifact-preview" tabIndex={0}>
                {source}
              </pre>
            ),
          },
        ]}
      />
    </div>
  );
}
