import { lazy, Suspense, useEffect } from "react";
import { Link, useLocation } from "react-router";

const MarkdownPreview = lazy(() => import("../../artifacts/previews/markdown"));

export function AuditMarkdown({ source }: { source: string }) {
  return (
    <div className="audit-markdown">
      <Suspense fallback={<p className="loading-copy">Loading text…</p>}>
        <MarkdownPreview source={source} />
      </Suspense>
    </div>
  );
}

export function AuditAnchor({ ready = true }: { ready?: boolean }) {
  const { hash } = useLocation();
  useEffect(() => {
    if (ready && hash !== "") {
      document
        .getElementById(hash.slice(1))
        ?.scrollIntoView({ block: "start" });
    }
  }, [hash, ready]);
  return null;
}

export function ProjectAuditNavigation({
  projectId,
  current,
}: {
  projectId: string;
  current: "audits" | "findings";
}) {
  return (
    <nav
      className="audit-section-navigation"
      aria-label="Project audit sections"
    >
      {(["audits", "findings"] as const).map((section) => (
        <Link
          key={section}
          to={`/projects/${encodeURIComponent(projectId)}/${section}`}
          className={current === section ? "active" : ""}
          aria-current={current === section ? "page" : undefined}
        >
          {section === "audits" ? "Audits" : "Findings"}
        </Link>
      ))}
    </nav>
  );
}
