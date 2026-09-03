import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MarkdownArtifactPreview({
  source,
}: {
  source: string;
}) {
  return (
    <article className="markdown-artifact-preview">
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ children, href, title }) => (
            <a
              href={href}
              rel="noreferrer noopener"
              target="_blank"
              title={title}
            >
              {children}
            </a>
          ),
          img: ({ alt }) => (
            <span className="markdown-image-placeholder">
              {alt === undefined || alt === ""
                ? "Image omitted"
                : `Image omitted: ${alt}`}
            </span>
          ),
        }}
      >
        {source}
      </Markdown>
    </article>
  );
}
