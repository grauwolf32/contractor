import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import {
  AUTOMATIC_OPENAPI_LIMIT_LABEL,
  createArtifactPreviewPlan,
  type ArtifactPreviewPlan,
} from "./preview-plan";
import { Button } from "../../ui";
import { SourcePreview } from "./source-preview";
import { ScanReportSummary } from "../../widgets/scan-reports/view";
import { PanelTabs } from "./tabs";
import "./materials.css";

const MarkdownArtifactPreview = lazy(() => import("./previews/markdown"));
const OpenApiArtifactPreview = lazy(() => import("./previews/openapi"));
const LikeC4ArtifactPreview = lazy(() => import("./previews/likec4"));
const DiffArtifactPreview = lazy(() => import("./previews/diff"));
const HTTPExchangePreview = lazy(() => import("./previews/http-exchange"));
const SitemapPreview = lazy(() => import("./previews/sitemap"));
const EngagementPreview = lazy(() => import("./previews/engagement"));

const RENDERER_LABELS = {
  diff: "Diff",
  markdown: "Markdown",
  openapi: "OpenAPI",
  "openapi-on-request": "OpenAPI",
  likec4: "LikeC4",
  http: "HTTP exchange",
  "invalid-http": "HTTP exchange",
  sitemap: "Sitemap",
  "invalid-sitemap": "Sitemap",
  engagement: "Engagement",
  "invalid-engagement": "Engagement",
} as const;

type RenderedPlan = Exclude<ArtifactPreviewPlan, { kind: "source" }>;

function Renderer({ plan, source }: { plan: RenderedPlan; source: string }) {
  switch (plan.kind) {
    case "engagement":
      return <EngagementPreview document={plan.document} />;
    case "invalid-engagement":
      return (
        <p className="artifact-renderer-error" role="alert">
          This engagement is invalid or unsupported. Source and download retain
          the original saved document.
        </p>
      );
    case "sitemap":
      return <SitemapPreview document={plan.document} />;
    case "invalid-sitemap":
      return (
        <p className="artifact-renderer-error" role="alert">
          This sitemap is invalid or unsupported. Source and download retain the
          original saved document.
        </p>
      );
    case "http":
      return <HTTPExchangePreview document={plan.document} />;
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
    case "openapi-on-request":
      return <OpenApiOnRequest source={source} />;
    case "likec4":
      return <LikeC4ArtifactPreview source={source} />;
  }
}

type OpenApiRequest =
  | { source: string; state: "reading" | "invalid" | "failed" }
  | { source: string; state: "ready"; document: Record<string, unknown> };

/**
 * An OpenAPI document larger than LIM-UI-17 is parsed and rendered only when
 * asked for (UI-R213): rendering every operation can hold the page.
 */
function OpenApiOnRequest({ source }: { source: string }) {
  const [request, setRequest] = useState<OpenApiRequest>();
  const current = request?.source === source ? request : undefined;

  useEffect(() => {
    if (current?.state !== "reading") {
      return;
    }
    let active = true;
    import("./previews/openapi-document").then(
      ({ parseOpenApiDocument }) => {
        const document = parseOpenApiDocument(source);
        if (active) {
          setRequest(
            document === undefined
              ? { source, state: "invalid" }
              : { source, state: "ready", document },
          );
        }
      },
      () => {
        if (active) {
          setRequest({ source, state: "failed" });
        }
      },
    );
    return () => {
      active = false;
    };
  }, [current?.state, source]);

  if (current?.state === "ready") {
    return <OpenApiArtifactPreview document={current.document} />;
  }
  if (current?.state === "invalid") {
    return (
      <p className="artifact-renderer-error" role="alert">
        This document is not a valid OpenAPI document. Source shows it as text.
      </p>
    );
  }
  return (
    <div className="artifact-renderer-request">
      <p className="muted-copy">
        This OpenAPI document is larger than {AUTOMATIC_OPENAPI_LIMIT_LABEL}, so
        it is not rendered automatically. Rendering it can make the page stop
        responding for a few seconds; Source shows it as text.
      </p>
      {current?.state === "failed" ? (
        <p className="artifact-renderer-error" role="alert">
          The renderer could not be loaded. Try again.
        </p>
      ) : null}
      <Button
        pending={current?.state === "reading"}
        pendingLabel="Rendering…"
        onClick={() => setRequest({ source, state: "reading" })}
      >
        Render document
      </Button>
    </div>
  );
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
}: {
  mediaType: string;
  source: string;
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
        {plan.memorySnapshot ? (
          <p className="ui-caption">
            Statements from memories · not independently verified. Names,
            content, descriptions and tags below are the saved originals.
          </p>
        ) : null}
        {mediaType === "application/json" ? (
          <ScanReportSummary source={source} />
        ) : null}
        <SourcePreview source={source} />
      </div>
    );
  }

  return (
    <div className="materials-preview-shell">
      <PanelTabs
        label="Preview mode"
        note={RENDERER_LABELS[plan.kind]}
        initial={plan.kind === "openapi-on-request" ? "source" : undefined}
        tabs={[
          {
            id: "rendered",
            label: [
              "sitemap",
              "invalid-sitemap",
              "engagement",
              "invalid-engagement",
            ].includes(plan.kind)
              ? "Preview"
              : "Rendered",
            panel: (
              <RendererErrorBoundary>
                <Suspense
                  fallback={
                    <p className="artifact-renderer-loading">
                      Loading renderer…
                    </p>
                  }
                >
                  <Renderer plan={plan} source={source} />
                </Suspense>
              </RendererErrorBoundary>
            ),
          },
          {
            id: "source",
            label: "Source",
            panel: <SourcePreview source={source} />,
          },
        ]}
      />
    </div>
  );
}
