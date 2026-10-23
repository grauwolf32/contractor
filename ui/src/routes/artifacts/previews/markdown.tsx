import {
  Fragment,
  memo,
  startTransition,
  useEffect,
  useMemo,
  useState,
} from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { splitFrontmatter } from "./frontmatter";
import {
  MAXIMUM_RENDERED_SECTION_CHARACTERS,
  splitMarkdownSections,
} from "./markdown-sections";
import "../reader.css";

const REMARK_PLUGINS = [remarkGfm];

// Sections rendered per later step. Measured in Chromium on a 1 MiB report:
// four finish it in about 2 s with no task much over 50 ms; one took 2.7 s,
// eight 1.8 s with 70 ms tasks.
const SECTIONS_PER_STEP = 4;

const COMPONENTS: Components = {
  a: ({ children, href, title }) => (
    <a href={href} rel="noreferrer noopener" target="_blank" title={title}>
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
};

const MarkdownSection = memo(function MarkdownSection({
  source,
}: {
  source: string;
}) {
  if (source.length > MAXIMUM_RENDERED_SECTION_CHARACTERS) {
    return (
      <>
        <p className="muted-copy">
          This part has more than{" "}
          {MAXIMUM_RENDERED_SECTION_CHARACTERS.toLocaleString("en-US")}{" "}
          characters without a break and is too long to render, so it is shown
          as text.
        </p>
        <pre tabIndex={0}>{source}</pre>
      </>
    );
  }
  return (
    <Markdown remarkPlugins={REMARK_PLUGINS} skipHtml components={COMPONENTS}>
      {source}
    </Markdown>
  );
});

/**
 * Renders the first section of the body at once and the others a few at a
 * time in later tasks (UI-R202), so a large document neither blocks the page
 * nor renders differently from a whole-document render.
 */
export default function MarkdownArtifactPreview({
  source,
}: {
  source: string;
}) {
  const document = useMemo(() => splitFrontmatter(source), [source]);
  const sections = useMemo(
    () => splitMarkdownSections(document.body),
    [document.body],
  );
  const [progress, setProgress] = useState({ sections, shown: 1 });
  if (progress.sections !== sections) {
    setProgress({ sections, shown: 1 });
  }
  const shown = progress.sections === sections ? progress.shown : 1;

  useEffect(() => {
    if (shown >= sections.length) {
      return;
    }
    const timer = setTimeout(() => {
      startTransition(() => {
        setProgress((current) =>
          current.sections === sections
            ? { sections, shown: current.shown + SECTIONS_PER_STEP }
            : current,
        );
      });
    }, 0);
    return () => clearTimeout(timer);
  }, [sections, shown]);

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
      {sections.slice(0, shown).map((section, index) => (
        // A whole-document render puts a line break between top-level blocks.
        <Fragment key={index}>
          {index === 0 ? null : "\n"}
          <MarkdownSection source={section} />
        </Fragment>
      ))}
      {shown < sections.length ? (
        <p className="muted-copy" role="status">
          Rendering the rest of the document…
        </p>
      ) : null}
    </article>
  );
}
