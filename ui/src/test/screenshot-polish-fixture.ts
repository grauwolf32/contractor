// Read-only synthetic API used by production-route browser review and its preview.
import { findingNeedsReview } from "../api/audits";
import {
  artifactMetadata,
  findingSourcePreview,
  findingSummary,
  report,
  conforming,
} from "./builders";
import { runStatus, stageAttempt } from "./builders-execution";
import { issueReviewFixtures } from "./issue-review-fixtures";
import { costFixture } from "./llm-cost-fixtures";
import { ownerListResponse } from "./owner-lists";
import {
  bytes,
  emptyPageOrNotFound,
  failure,
  json,
  page,
  unavailableCostForURL,
} from "./responses";

export function screenshotPolishFixture(
  options: {
    partialInbox?: boolean;
    cost?: "missing" | "partial" | "error";
  } = {},
) {
  const data = issueReviewFixtures();
  const project = data.projects[0]!;
  project.name = "Invoice Lab";
  project.description =
    "Security review of the invoice API and its supporting materials.";
  const now = new Date().toISOString();
  const started = new Date(Date.parse(now) - 300_000).toISOString();
  for (const audit of data.audits) {
    audit.updatedAt = now;
    audit.finishedAt = now;
  }
  const pendingIssue = data.findings.find(
    (f) => f.projectId === project.projectId && findingNeedsReview(f),
  )!;
  const savedIssue = data.findings.find((f) => f.analystVerdict !== null)!;
  const contents: Record<string, string> = {
    "outputs/report":
      "# Invoice API review\n\nThis retained result contains the recorded observations.\n\n- Review evidence\n- Confirm the affected boundary",
    "openapi/invoice-api":
      "openapi: 3.0.3\ninfo:\n  title: Invoice API\n  version: '1.0'\npaths:\n  /invoices:\n    get:\n      responses:\n        '200':\n          description: Invoice list\n",
    "outputs/template_manifest":
      '{"templates":["invoice-exposure"],"source":"synthetic preview"}\n',
  };
  const output = artifactMetadata("outputs", "report", {
    artifact: {
      namespace: "outputs",
      name: "report",
      revision: "rev_b551a66477ecc8cb0083d3184723cc4b",
    },
    mediaType: "text/markdown",
    size: new TextEncoder().encode(contents["outputs/report"]!).byteLength,
  });
  const materials = [
    artifactMetadata("openapi", "invoice-api", {
      mediaType: "application/yaml",
    }),
    output,
    artifactMetadata("outputs", "template_manifest", {
      mediaType: "application/vnd.contractor.nuclei-manifest+json",
    }),
  ];
  for (const material of materials) {
    material.size = new TextEncoder().encode(
      contents[`${material.artifact.namespace}/${material.artifact.name}`]!,
    ).byteLength;
    material.createdAt = now;
  }
  const attempt = stageAttempt("stage_review_failed", {
    stage: "review",
    state: "failed",
    objective: "Review the invoice API within its declared engagement.",
    createdAt: now,
    updatedAt: now,
    terminalAt: now,
    termination: {
      outcome: "interrupted",
      code: "worker_deadline_exceeded",
      message: "Worker invocation deadline expired",
      retryable: true,
      phase: "running",
      occurredAt: now,
    },
    diagnostics: {
      truncated: false,
      items: [
        {
          participant: "planner",
          code: "worker_deadline_exceeded",
          message: "Worker invocation deadline expired",
          retryable: true,
        },
        {
          participant: "worker",
          logicalAgent: "analyst",
          code: "worker_cancelled",
          message: "The Worker stopped after the invocation deadline.",
        },
      ],
    },
  });
  const runs = [
    runStatus("run_result_preview", {
      projectId: project.projectId,
      workflow: "invoice-review@3",
      state: "succeeded",
      createdAt: started,
      finishedAt: now,
      updatedAt: now,
      outputs: { report: output.artifact },
    }),
    runStatus("run_failed_preview", {
      projectId: project.projectId,
      workflow: "invoice-review@3",
      state: "failed",
      createdAt: started,
      finishedAt: now,
      updatedAt: now,
      attempts: [attempt],
    }),
  ];
  const summaries = runs.map(
    ({
      runId,
      projectId,
      workflow,
      state,
      deletable,
      labels,
      createdAt,
      updatedAt,
      finishedAt,
    }) =>
      conforming("RunSummary", {
        runId,
        ...(projectId === undefined ? {} : { projectId }),
        workflow,
        state,
        deletable,
        labels,
        createdAt: createdAt ?? now,
        updatedAt: updatedAt ?? now,
        ...(finishedAt === undefined ? {} : { finishedAt }),
      }),
  );
  const callReference =
    "8d48b8da27677a842717c1061278ac439bb619ae5e62ca61c739ca8024ed0e25";
  const paths = {
    inbox: "/",
    issue: `/issues/${savedIssue.projectId}/${savedIssue.findingId}?state=all`,
    overview: `/projects/${project.projectId}`,
    materials: `/projects/${project.projectId}/artifacts`,
    run: "/runs/run_failed_preview",
    report: `/?item=report:${data.audits[0]!.auditId}`,
  };
  function respond(request: Request, url = new URL(request.url)): Response {
    const path = url.pathname;
    if (request.method !== "GET")
      return failure(
        403,
        "forbidden",
        "Read-only preview; no business data is changed.",
      );
    const cost = unavailableCostForURL(request.method, url);
    if (cost) {
      if (options.cost === "error")
        return failure(
          503,
          "storage_transaction_conflict",
          "Cost storage is temporarily unavailable.",
        );
      if (options.cost === "partial")
        return json(
          costFixture(cost.scope, cost.scopeId, {
            coverage: "partial",
            historicalGap: true,
            unknownReports: "2",
          }),
        );
      return json(cost);
    }
    if (path === "/v1/projects") return json(page(data.projects));
    if (path === "/v1/queue/control")
      return json({ paused: false, revision: "0" }, 200, { ETag: '"0"' });
    if (options.partialInbox && path === "/v1/runs")
      return failure(503, "storage_transaction_conflict");
    const owned = ownerListResponse(url, data);
    if (owned !== undefined) return owned;
    const exactProject = data.projects.find(
      (p) => path === `/v1/projects/${p.projectId}`,
    );
    if (exactProject)
      return json(exactProject, 200, { ETag: `"${exactProject.revision}"` });
    const audit = data.audits.find((a) => path === `/v1/audits/${a.auditId}`);
    if (audit) return json(audit, 200, { ETag: '"1"' });
    if (path.endsWith("/report") && path.startsWith("/v1/audits/"))
      return json(
        report(path.split("/")[3]!, "ready", {
          summary:
            "# Invoice API review\n\n## Recorded results\n\nTwo possible issues require human review. Supporting evidence is available in the check.\n\nThis report is not a security or compliance certification.",
        }),
      );
    const finding = data.findings.find(
      (f) => path === `/v1/projects/${f.projectId}/findings/${f.findingId}`,
    );
    if (finding) return json(finding, 200, { ETag: `"${finding.revision}"` });
    const source = data.sources.find(
      (s) =>
        path === `/v1/projects/${s.projectId}/finding-sources/${s.receiptId}`,
    );
    if (source) return json(source);
    const sourceFinding = data.findings.find(
      (f) =>
        path === `/v1/projects/${f.projectId}/findings/${f.findingId}/sources`,
    );
    if (sourceFinding)
      return json({
        ...page(
          data.sources
            .filter((s) => s.findingId === sourceFinding.findingId)
            .map(findingSourcePreview),
        ),
        projectRevision: 1,
      });
    if (path.endsWith("/history"))
      return json({ ...page([]), projectRevision: 1 });
    if (path === `/v1/projects/${project.projectId}/findings`)
      return json({
        ...page(
          data.findings
            .filter((f) => f.projectId === project.projectId)
            .map(findingSummary),
        ),
        projectRevision: 1,
      });
    if (path === `/v1/projects/${project.projectId}/artifacts`)
      return json(
        page(
          materials.filter(
            (m) =>
              !url.searchParams.has("namespace") ||
              m.artifact.namespace === url.searchParams.get("namespace"),
          ),
        ),
      );
    const material = materials.find(
      (m) =>
        path ===
        `/v1/projects/${project.projectId}/artifacts/${m.artifact.namespace}/${m.artifact.name}/metadata`,
    );
    if (material) return json(material);
    const materialBytes = materials.find(
      (m) =>
        path ===
        `/v1/projects/${project.projectId}/artifacts/${m.artifact.namespace}/${m.artifact.name}`,
    );
    if (materialBytes)
      return bytes(
        contents[
          `${materialBytes.artifact.namespace}/${materialBytes.artifact.name}`
        ]!,
        materialBytes.mediaType,
      );
    if (
      path === "/v1/runs" ||
      path === `/v1/projects/${project.projectId}/runs`
    )
      return json(
        page(
          summaries.filter(
            (r) =>
              !url.searchParams.has("state") ||
              url.searchParams.get("state")!.split(",").includes(r.state),
          ),
        ),
      );
    const run = runs.find((r) => path === `/v1/runs/${r.runId}`);
    if (run) return json(run);
    if (path.endsWith("/tool-calls"))
      return json({
        status: "available",
        participant: url.searchParams.get("participant") ?? "planner",
        ...(url.searchParams.has("logicalAgent")
          ? { logicalAgent: url.searchParams.get("logicalAgent") }
          : {}),
        complete: true,
        detailTruncated: false,
        items: [
          {
            callReference,
            tool: "telemetry.export",
            toolTruncated: false,
            outcome: "failed",
            durationMs: null,
            resultSizeBytes: null,
            argumentsTruncated: false,
            error: {
              code: "flush_timeout",
              message: "Planner telemetry export failed",
              messageTruncated: false,
              retryable: false,
            },
          },
        ],
        page: { hasMore: false },
      });
    if (path.endsWith("/runtime-diagnostics"))
      return json({ status: "available", items: [], page: { hasMore: false } });
    if (path.startsWith("/v1/workflows/invoice-review/"))
      return failure(
        404,
        "not_found",
        "The exact Workflow definition is unavailable.",
      );
    if (path.includes("/artifacts/outputs/report"))
      return path.endsWith("/metadata")
        ? json(output)
        : bytes(contents["outputs/report"]!, "text/markdown");
    const evidence = data.evidence.find((e) =>
      path.startsWith(
        `/v1/projects/${e.projectId}/artifacts/${e.metadata.artifact.namespace}/${e.metadata.artifact.name}`,
      ),
    );
    if (evidence)
      return path.endsWith("/metadata")
        ? json(evidence.metadata)
        : bytes(evidence.text, evidence.metadata.mediaType);
    return emptyPageOrNotFound(request, url) ?? failure(404);
  }
  return {
    data,
    project,
    savedIssue,
    pendingIssue,
    runs,
    callReference,
    paths,
    respond,
  };
}
