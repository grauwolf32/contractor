import { lazy, Suspense, useEffect } from "react";
import { Link, useLocation } from "react-router";

import type { Audit } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { exactArtifactLink } from "./artifact-links";

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
      className="audit-section-navigation section-navigation"
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

export function ExactArtifactLink({
  projectId,
  artifact,
  label,
  projectReadable = false,
}: {
  projectId: string;
  artifact: Audit["inputs"][string];
  label?: string;
  projectReadable?: boolean;
}) {
  const content = (
    <>
      {label === undefined ? null : <strong>{label}</strong>}
      <code>
        {artifact.ref.namespace}/{artifact.ref.name}@{artifact.ref.revision}
      </code>
    </>
  );
  return projectReadable ? (
    <ContextLink
      returnLabel="Audit"
      className="artifact-ref-link"
      to={exactArtifactLink(projectId, artifact)}
      title={artifact.digest}
    >
      {content}
    </ContextLink>
  ) : (
    <span className="artifact-ref-link" title={artifact.digest}>
      {content}
    </span>
  );
}
