import { findingNeedsReview, findingSeverity } from "../api/audits";
import {
  artifactMetadata,
  finding,
  findingSource,
  findingSourcePreview,
  findingSummary,
  project,
} from "./builders";
import { bytes, failure, json, page } from "./responses";

/** One canonical backend with 10,000 compact records and two source receipts. */
export function projectFactsFixture(total = 10_000) {
  const owner = project("project_facts", { name: "Facts project" });
  const fact = finding(
    owner.projectId,
    "fact_nginx",
    {},
    {
      title: "Reported nginx deployment",
      description: "A memory reports nginx. No independent verification.",
      severity_suggestion: "informational",
    },
  );
  const ordinary = finding(
    owner.projectId,
    "info_security",
    {},
    {
      title: "Informational security observation",
      severity_suggestion: "informational",
    },
  );
  const unknown = finding(
    owner.projectId,
    "unknown_severity",
    {},
    { title: "Unknown severity record", severity_suggestion: "" },
  );
  const low = finding(
    owner.projectId,
    "low_severity",
    {},
    { title: "Low severity record", severity_suggestion: "low" },
  );
  const lowSummary = findingSummary(low);
  const summaries = Array.from({ length: total }, (_, i) => ({
    ...lowSummary,
    findingId: `record_${i}`,
    title: `Other record ${i}`,
  }));
  summaries[0] = findingSummary(fact);
  summaries[1] = findingSummary(unknown);
  summaries[2] = findingSummary(low);
  summaries[Math.min(300, total - 1)] = findingSummary(ordinary);
  const original = JSON.stringify({
    schemaVersion: "contractor.memory-snapshot/v1",
    namespace: "research",
    notes: [
      {
        name: "service_banner",
        content:
          "nginx reported by the banner; version not verified. <script>window.memoryExecuted=true</script>",
        description: "Original technology memory",
        tags: ["reported"],
      },
    ],
  });
  const metadata = artifactMetadata("finding-materials", "original-memory", {
    artifact: {
      namespace: "finding-materials",
      name: "original-memory",
      revision: "memory-r1",
    },
    mediaType: "application/json",
    size: new TextEncoder().encode(original).length,
    frozen: true,
  });
  const evidence = {
    ref: metadata.artifact,
    digest: fact.firstProposal.digest,
    mediaType: metadata.mediaType,
    sizeBytes: metadata.size,
  };
  const first = findingSource(
    fact,
    { evidence: [evidence], projectEvidence: [evidence] },
    { evidence_ids: ["original-memory"] },
  );
  const second = findingSource(
    fact,
    {
      receiptId: "receipt_independent",
      proposalId: "proposal_independent",
      clientKey: "independent",
      evidence: [evidence],
      projectEvidence: [evidence],
      origin: {
        ...first.origin,
        workflow: { ...first.origin.workflow, name: "independent-memory" },
      },
    },
    {
      client_key: "independent",
      title: "Another title over the same memory",
      description:
        "Independent reported inference, without a Task or hypothesis.",
      evidence_ids: ["original-memory"],
    },
  );
  const listReads: number[] = [],
    payloadReads: string[] = [],
    sourceReads: string[] = [];
  let unavailable = false;
  function assess() {
    Object.assign(fact, {
      analystSeverity: "high",
      analystVerdict: "true_positive",
      state: "confirmed",
      revision: 2,
      actorId: "analyst",
      rationale: "Explicit analyst assessment",
      assessedAt: fact.createdAt,
    });
    summaries[0] = findingSummary(fact);
  }
  function paged(url: URL, items: unknown[]) {
    const offset = Number(url.searchParams.get("cursor") ?? 0),
      limit = Number(url.searchParams.get("limit") ?? 50);
    return {
      items: items.slice(offset, offset + limit),
      page: {
        hasMore: offset + limit < items.length,
        ...(offset + limit < items.length
          ? { nextCursor: String(offset + limit) }
          : {}),
      },
    };
  }
  const respond = (request: Request): Response | undefined => {
    const url = new URL(request.url),
      path = url.pathname,
      root = `/v1/projects/${owner.projectId}`;
    if (request.method !== "GET") return failure(403, "forbidden");
    if (path === "/v1/projects") return json(page([owner]));
    if (path === root) return json(owner, 200, { ETag: '"1"' });
    if (path === `${root}/findings`) {
      listReads.push(Number(url.searchParams.get("cursor") ?? 0));
      return json({ ...paged(url, summaries), projectRevision: 1 });
    }
    if (path === "/v1/findings") {
      const values = summaries.filter((row) => {
        const severity = url.searchParams.get("severity"),
          state = url.searchParams.get("state"),
          verdict = url.searchParams.get("verdict");
        return (
          (!severity || severity.split(",").includes(findingSeverity(row))) &&
          (!state || state.split(",").includes(row.state ?? "")) &&
          (!verdict ||
            (verdict === "needs_review"
              ? findingNeedsReview(row)
              : verdict.split(",").includes(row.analystVerdict ?? "")))
        );
      });
      return json(paged(url, values));
    }
    if (path.startsWith(`${root}/findings/`)) {
      const id = path.slice(`${root}/findings/`.length);
      const known = [fact, ordinary, unknown, low].find(
        (row) => row.findingId === id,
      );
      return json(known ?? finding(owner.projectId, id), 200, { ETag: '"1"' });
    }
    if (path === `${root}/finding-sources`) {
      sourceReads.push(url.toString());
      return json({
        items: [
          findingSourcePreview(url.searchParams.has("cursor") ? second : first),
        ],
        projectRevision: 1,
        page: url.searchParams.has("cursor")
          ? { hasMore: false }
          : { hasMore: true, nextCursor: "second-source" },
      });
    }
    const source = [first, second].find(
      (value) => path === `${root}/finding-sources/${value.receiptId}`,
    );
    if (source) return json(source);
    if (path.endsWith("/history"))
      return json({ ...page([]), projectRevision: 1 });
    const artifactPath = `${root}/artifacts/${metadata.artifact.namespace}/${metadata.artifact.name}`;
    if (path === artifactPath || path === `${artifactPath}/metadata`) {
      if (url.searchParams.get("revision") !== "memory-r1")
        return failure(404, "not_found");
      if (unavailable)
        return failure(404, "not_found", "Original memory is unavailable");
      if (path.endsWith("/metadata")) return json(metadata);
      payloadReads.push(url.toString());
      return bytes(original, "application/json");
    }
    return undefined;
  };
  return {
    owner,
    fact,
    ordinary,
    unknown,
    low,
    first,
    second,
    original,
    metadata,
    listReads,
    payloadReads,
    sourceReads,
    respond,
    assess,
    setUnavailable(value: boolean) {
      unavailable = value;
    },
  };
}
