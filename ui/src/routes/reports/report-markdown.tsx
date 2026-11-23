import type { ReactNode } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// A summary sits under the page's own headings (the "Summary" h3), so its
// headings start at h4; `data-level` keeps the level the author wrote for
// styling. As in the Markdown preview of materials, raw HTML is skipped,
// images are omitted and links open in a new tab without a referrer.
function heading(level: 1 | 2 | 3 | 4 | 5 | 6) {
  return function Heading({ children }: { children?: ReactNode }) {
    if (level === 1) return <h4 data-level={level}>{children}</h4>;
    if (level === 2) return <h5 data-level={level}>{children}</h5>;
    return <h6 data-level={level}>{children}</h6>;
  };
}

const COMPONENTS: Components = {
  h1: heading(1),
  h2: heading(2),
  h3: heading(3),
  h4: heading(4),
  h5: heading(5),
  h6: heading(6),
  a: ({ children, href, title }) => (
    <a href={href} rel="noreferrer noopener" target="_blank" title={title}>
      {children}
    </a>
  ),
  img: ({ alt }) => (
    <span className="reports-markdown-image">
      {alt === undefined || alt === ""
        ? "Image omitted"
        : `Image omitted: ${alt}`}
    </span>
  ),
};

/** A report summary rendered as Markdown. Loaded lazily with its parser. */
export default function ReportMarkdown({ source }: { source: string }) {
  return (
    <div className="reports-markdown">
      <Markdown remarkPlugins={[remarkGfm]} skipHtml components={COMPONENTS}>
        {source}
      </Markdown>
    </div>
  );
}
