import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import type { ArtifactMetadata } from "../api/artifacts";
import {
  getArtifactArchive,
  getArtifactArchiveFile,
} from "../api/artifact-archive";
import { usePublicAPI } from "../api/context";
import { queryKeys } from "../api/query-keys";
import { plural } from "../app/format";
import { Button } from "../ui";
import { splitFrontmatter } from "./artifacts/previews/frontmatter";

/**
 * A Skill package's purpose: the `description` of its SKILL.md front matter
 * and its file count, read once the row scrolls into view.
 */
export function SkillDescription({ metadata }: { metadata: ArtifactMetadata }) {
  const api = usePublicAPI();
  const element = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (element.current === null || typeof IntersectionObserver === "undefined")
      return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    });
    observer.observe(element.current);
    return () => observer.disconnect();
  }, []);
  const query = useQuery({
    queryKey: queryKeys.catalog.skillDescription(metadata.artifact),
    enabled: visible && metadata.size <= 128 * 1024,
    retry: false,
    staleTime: Infinity,
    queryFn: async ({ signal }) => {
      const archive = await getArtifactArchive(
        api,
        { kind: "user" },
        metadata.artifact,
        signal,
      );
      const files = archive.entries.filter((entry) => entry.kind === "file");
      const entry = files.find((file) => file.path === "SKILL.md");
      if (entry?.previewable !== true || entry.size > 64 * 1024)
        return { count: files.length };
      const file = await getArtifactArchiveFile(
        api,
        { kind: "user" },
        metadata.artifact,
        entry.path,
        signal,
      );
      return {
        count: files.length,
        description: splitFrontmatter(file.text).fields.description,
      };
    },
  });
  if (query.isError) {
    return (
      <div ref={element} className="skill-description-error">
        <p className="skill-description" role="status">
          Description unavailable
        </p>
        <Button
          size="xs"
          pending={query.isFetching}
          pendingLabel="Retrying…"
          onClick={() => void query.refetch()}
        >
          Retry
        </Button>
      </div>
    );
  }
  return (
    <div ref={element} className="skill-purpose">
      <p className="skill-description" title={query.data?.description}>
        {query.data?.description ??
          (query.isFetching
            ? "Reading description…"
            : "Open the package to inspect its instructions.")}
      </p>
      {query.data === undefined ? null : (
        <small className="skill-files">
          {plural(query.data.count, "file")}
        </small>
      )}
    </div>
  );
}
