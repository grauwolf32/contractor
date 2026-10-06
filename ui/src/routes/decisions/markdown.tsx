import { lazy, Suspense } from "react";

// The Markdown renderer the check pages use for finding descriptions and
// decision reasons (raw HTML skipped, images omitted, links open safely).
const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));

/** The AI's or a reviewer's Markdown text, styled for the decision panes. */
export function DecisionMarkdown({ source }: { source: string }) {
  return (
    <div className="decisions-markdown">
      <Suspense fallback={<p className="decisions-quiet">Loading text…</p>}>
        <MarkdownPreview source={source} />
      </Suspense>
    </div>
  );
}
