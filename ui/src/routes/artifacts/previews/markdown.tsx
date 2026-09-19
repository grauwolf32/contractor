import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { splitFrontmatter } from "./frontmatter";

export default function MarkdownArtifactPreview({
  source,
}: {
  source: string;
}) {
  const document = splitFrontmatter(source);
  return (
    <article className="markdown-artifact-preview">
      {Object.keys(document.fields).length === 0 ? null : (
        <dl className="document-frontmatter">
          {Object.entries(document.fields).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {document.invalid ? (
        <p className="muted-copy">
          Document metadata could not be read. The original is available in
          Source.
        </p>
      ) : null}
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
        {document.body}
      </Markdown>
    </article>
  );
}
