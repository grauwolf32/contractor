import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import type { ProjectFinding } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import {
  downloadFindingExport,
  MAX_EXPORT_SELECTION,
  publishFindingExport,
  type FindingExportPart,
  type FindingExportRequest,
  type PublishedFindingExport,
} from "../../../api/finding-exports";
import { saveBlob } from "../../../app/download";

type Selection = FindingExportRequest["findings"][number];

/** One mounted check owns its explicit revision selection and retry key. */
export function useFindingExport(auditId: string) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Map<string, Selection>>(new Map());
  const [includeDuplicates, setIncludeDuplicates] = useState(false);
  const [published, setPublished] = useState<PublishedFindingExport>();
  const attempt = useRef<FindingExportRequest | undefined>(undefined);
  const active = useRef<AbortController | undefined>(undefined);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      active.current?.abort();
    };
  }, []);
  const publication = useMutation({
    retry: false,
    mutationFn: (request: FindingExportRequest) => {
      active.current = new AbortController();
      return publishFindingExport(api, auditId, request, active.current.signal);
    },
    onSuccess: (value) => {
      if (alive.current) setPublished(value);
    },
  });
  const download = useMutation({
    retry: false,
    mutationFn: (part: FindingExportPart) => {
      if (published === undefined) throw new Error("Create an export first");
      active.current = new AbortController();
      return downloadFindingExport(
        api,
        auditId,
        published,
        part,
        active.current.signal,
      );
    },
    onSuccess: (value) => {
      if (alive.current) saveBlob(value.blob, value.filename);
    },
  });
  function changed() {
    attempt.current = undefined;
    setPublished(undefined);
    publication.reset();
    download.reset();
  }
  function selectVisible(findings: readonly ProjectFinding[]) {
    if (
      new Set([...selected.keys(), ...findings.map((f) => f.findingId)]).size >
      MAX_EXPORT_SELECTION
    )
      return;
    const next = new Map(selected);
    for (const finding of findings) {
      // Existing selections keep their chosen revision until explicitly removed.
      if (!next.has(finding.findingId))
        next.set(finding.findingId, {
          findingId: finding.findingId,
          revision: finding.revision,
        });
    }
    if (next.size === selected.size) return;
    changed();
    setSelected(next);
  }
  return {
    open,
    setOpen,
    selected,
    includeDuplicates,
    published,
    publication,
    download,
    busy: publication.isPending || download.isPending,
    clear() {
      changed();
      setSelected(new Map());
    },
    toggle(finding: ProjectFinding) {
      const next = new Map(selected);
      if (next.has(finding.findingId)) next.delete(finding.findingId);
      else if (next.size < MAX_EXPORT_SELECTION)
        next.set(finding.findingId, {
          findingId: finding.findingId,
          revision: finding.revision,
        });
      if (next.size === selected.size) return;
      changed();
      setSelected(next);
    },
    selectVisible,
    changeDuplicates(value: boolean) {
      if (value === includeDuplicates) return;
      changed();
      setIncludeDuplicates(value);
    },
    publish() {
      if (attempt.current === undefined)
        attempt.current = {
          clientKey: `ui-export-${crypto.randomUUID()}`,
          format: "sarif",
          profile: "generic",
          includeDuplicates,
          findings: [...selected.values()].sort((a, b) =>
            a.findingId < b.findingId ? -1 : a.findingId > b.findingId ? 1 : 0,
          ),
        };
      publication.mutate(attempt.current);
    },
  };
}
