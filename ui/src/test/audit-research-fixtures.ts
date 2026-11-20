// Synthetic current-contract data shared by browser, component and visual review.
import type { components } from "../api/generated/public";
import {
  audit,
  conforming,
  finding,
  findingSummary,
  project,
  workspace,
} from "./builders";
import { bytes, failure, json, page } from "./responses";

type Schemas = components["schemas"];
const at = "2026-11-06T10:20:00Z",
  digest = `sha256:${"a".repeat(64)}`;
export const RESEARCH_PROJECT = "project_shop",
  RESEARCH_AUDIT = "audit_research";

export function researchFixture(state = "normal") {
  let current = audit(
    RESEARCH_PROJECT,
    RESEARCH_AUDIT,
    state === "active" ? "active" : "paused",
    {
      profile: { name: "source-checklist", version: "1", digest },
      scope: {
        objective: "Can a customer read or export another customer’s orders?",
        target: "Shop API · order and invoice endpoints",
      },
      taskCount: 8,
      submittedRunCount: 5,
      outstandingRunCount: state === "active" ? 1 : 0,
      revision: 3,
      createdAt: at,
      updatedAt: at,
    },
  );
  const owner = project(RESEARCH_PROJECT, { name: "Shop API" });
  const subjects = [
    "A customer can read another customer’s order",
    "Product search input can change the SQL query",
    "A refresh token can be reused across sessions",
    "An invoice export can cross account boundaries",
    "A tenant can enumerate another tenant’s exports",
    "A session can be used after sign out",
    "A customer can access administrator functions",
  ];
  const materials: Schemas["AuditMaterialEntry"][] = [];
  const payloads = new Map<string, string>();
  function material(
    name: string,
    content: string,
    type = "text/markdown",
    key = "material/" + name,
  ) {
    const m = conforming("AuditTaskMaterial", {
      scope_kind: "audit",
      scope_id: RESEARCH_AUDIT,
      ref: { namespace: "retained", name, revision: name + "-r1" },
      digest,
      media_type: type,
      size_bytes: new TextEncoder().encode(content).byteLength,
    });
    const entry = { key, material: m, created_at: at };
    materials.push(entry);
    payloads.set(key, content);
    return m;
  }
  const observations = material(
    "order-api-observations",
    "# Order API observations\n\nOrder lookup and invoice exports accept a customer-supplied identifier.\n\nSelected baseline; two synthetic customers. No live target traffic.",
  );
  material(
    "shop-api",
    "openapi: 3.1.0\ninfo:\n  title: Shop API\n  version: '1'\npaths: {}",
    "application/yaml",
  );
  const longGround = material(
    "ground-" + "c".repeat(64),
    "Exact retained observations from a synthetic boundary.",
  );
  const hypotheses: Schemas["AuditHypothesis"][] = subjects.map(
    (statement, index) =>
      conforming("AuditHypothesis", {
        source_key: "assertion-" + index,
        fixed: [0, 1, 5, 6].includes(index),
        ordinal: index + 1,
        document: {
          schema: "contractor.audit.hypothesis.v1",
          id: "hypothesis_" + index,
          revision: 1,
          statement,
          grounds:
            index === 0
              ? "The order API accepts a customer-provided identifier. Ownership enforcement needs an observed comparison."
              : "The selected source describes a boundary that needs an explicit scoped check.",
          scope: {
            conditions: "Pinned Shop API baseline; customer A and customer B",
          },
          sources: index === 2 ? [observations, longGround] : [observations],
          origin: [0, 1, 5, 6].includes(index)
            ? { kind: "checklist", audit_id: RESEARCH_AUDIT }
            : {
                kind: "run",
                audit_id: RESEARCH_AUDIT,
                task_id: "task_3",
                run_id: "run_explore",
              },
        },
      }),
  );
  const verdicts = ["supported", "refuted", "inconclusive", undefined] as const;
  const rationales = [
    "Customer A received order_b42 owned by customer B. The recorded response included order contents and delivery details.",
    "The selected search route uses bound query parameters. Tested input did not change the query structure.",
    "The required independent second session was unavailable. Cross-session reuse could not be determined.",
    "The selected API exposes order lookup and invoice export. These exploratory observations do not require a hypothesis.",
  ];
  const titles = [
    "Order ownership check",
    "Product search query check",
    "Refresh-token reuse check",
    "Discover order API surface",
    "Admin export inspection",
    "Check invoice export access",
    "Compare tenant boundaries",
    "Review session isolation",
  ];
  const tasks: Schemas["AuditTask"][] = titles.map((objective, index) => {
    const h =
      index < 3
        ? hypotheses[index]
        : index === 5
          ? hypotheses[3]
          : index === 7
            ? hypotheses[5]
            : undefined;
    const id = "task_" + index;
    const result =
      index < 4
        ? JSON.stringify({
            schema: "contractor.audit.result.v1",
            items: [
              {
                item_key: "item-" + index,
                ...(verdicts[index] ? { verdict: verdicts[index] } : {}),
                rationale: rationales[index],
                tested_conditions: [
                  "Selected API baseline",
                  "Customer A and customer B",
                ],
                evidence: [observations],
                limitations: [
                  index === 2
                    ? "A second independent session is required"
                    : "Only the selected endpoint and baseline were tested",
                ],
                origin: {
                  audit_id: RESEARCH_AUDIT,
                  task_id: id,
                  run_id: index === 3 ? "run_explore" : "run_" + index,
                  ...(h
                    ? { hypothesis: { id: h.document.id, revision: 1 } }
                    : {}),
                },
              },
            ],
          })
        : undefined;
    return conforming("AuditTask", {
      attention: false,
      document: {
        schema: "contractor.audit.task.v1",
        audit_id: RESEARCH_AUDIT,
        task_id: id,
        workflow: index === 3 ? "explore" : "inspect",
        objective,
        inputs: { source: observations },
        parameters: {},
        ...(h ? { hypothesis: { id: h.document.id, revision: 1 } } : {}),
      },
      ...(h ? { hypothesis_snapshot: h.document } : {}),
      source: { kind: "initial", key: "work-" + index },
      ordinal: index + 1,
      iteration: 1,
      state: index < 5 ? "settled" : index === 5 ? "selected" : "ready",
      attempts: index < 6 ? 1 : 0,
      ...(result
        ? {
            result: material(
              "result-" + index,
              result,
              "application/json",
              `task/${id}/result`,
            ),
          }
        : {}),
      ...(index === 4 ? { gap_code: "proposal-ground-unavailable" } : {}),
      created_at: at,
      updated_at: at,
    });
  });
  const problem = finding(
    RESEARCH_PROJECT,
    "finding_order",
    {},
    {
      title: "Cross-account order details",
      description: "The selected customer received another customer’s order.",
      subject: { kind: "operation", key: "GET /api/orders/{id}" },
      severity_suggestion: "high",
    },
  );
  const program: Schemas["AuditProgram"] = {
    schema: "contractor.audit.program.v1",
    inputs: {},
    workflows: {
      inspect: {
        ref: "source-review@1",
        inputs: { source: { source: "task-input" } },
        outputs: { result: "result" },
        parameters: {},
      },
      explore: {
        ref: "research@1",
        inputs: {},
        outputs: { proposals: "proposals" },
        parameters: {},
      },
    },
    hypotheses: {},
    initial_work: [
      {
        key: "explore",
        workflow: "explore",
        objective: "Explore the selected API",
      },
    ],
    limits: {
      max_tasks: 128,
      max_tasks_per_iteration: 128,
      max_iterations: 4,
      max_submitted_runs: 128,
      max_attempts: 3,
      batch_size: 1,
      max_evidence_bytes: 1048576,
      deadline_seconds: 3600,
      max_hypotheses: 128,
      max_hypothesis_revisions: 512,
      max_hypothesis_bytes: 1048576,
    },
    policy: { active_operations: "prohibited" },
    continuation: { workflow: "explore" },
  };
  let hypothesisWrites = 0,
    publicationWrites = 0,
    taskWrites = 0;
  const calls: Array<{
    method: string;
    path: string;
    body?: unknown;
    key?: string | null;
  }> = [];
  async function respond(request: Request): Promise<Response | undefined> {
    const url = new URL(request.url),
      path = url.pathname,
      method = request.method;
    const call: {
      method: string;
      path: string;
      body?: unknown;
      key?: string | null;
    } = { method, path };
    calls.push(call);
    if (method !== "GET") {
      call.body = await request
        .clone()
        .json()
        .catch(() => undefined);
      call.key = request.headers.get("Idempotency-Key");
    }
    if (path === `/v1/projects/${RESEARCH_PROJECT}`)
      return json(owner, 200, { ETag: `"${owner.revision}"` });
    if (path === `/v1/projects/${RESEARCH_PROJECT}/audits`)
      return json(page([current]));
    if (path === `/v1/audit-profiles/source-checklist/versions/1`)
      return json(
        {
          ref: current.profile,
          inputs: {},
          limits: program.limits,
          policy: program.policy,
          program,
        },
        200,
        { ETag: `"${digest}"` },
      );
    if (path === `/v1/audits/${RESEARCH_AUDIT}`)
      return json(current, 200, { ETag: `"${current.revision}"` });
    if (path.endsWith("/pause") || path.endsWith("/resume")) {
      current = {
        ...current,
        revision: current.revision + 1,
        state: path.endsWith("/pause") ? "paused" : "active",
      };
      return json(current, 200, { ETag: `"${current.revision}"` });
    }
    if (path === `/v1/audits/${RESEARCH_AUDIT}/workspace`)
      return state === "stale-read"
        ? failure(503, "storage_transaction_conflict")
        : json(
            workspace(current, {
              totalTasks: 8,
              completedTasks: 4,
              gaps: 1,
              pendingTasks: 3,
              findings: 1,
              needsReviewFindings: 1,
            }),
          );
    if (path === `/v1/audits/${RESEARCH_AUDIT}/hypotheses`) {
      if (method === "POST") {
        hypothesisWrites++;
        if (state === "conflict" && hypothesisWrites === 1)
          return failure(412, "precondition_failed");
        const body = call.body as Schemas["AuditHypothesisContent"];
        const next = conforming("AuditHypothesis", {
          document: {
            schema: "contractor.audit.hypothesis.v1",
            id: "hypothesis_manual",
            revision: 1,
            statement: body.statement,
            scope: body.scope,
            grounds: body.grounds,
            origin: { kind: "manual", audit_id: current.auditId },
          },
          source_key: body.key,
          fixed: false,
          ordinal: 8,
        });
        hypotheses.push(next);
        current = { ...current, revision: current.revision + 1 };
        return json({
          hypotheses: [{ id: next.document.id, revision: 1 }],
          audit_revision: current.revision,
        });
      }
      return json({
        ...page(hypotheses),
        checking_workflows: ["inspect"],
        fixed_denominator: 4,
        pending_fixed_count: 0,
      });
    }
    if (path === `/v1/audits/${RESEARCH_AUDIT}/tasks`) {
      if (method === "POST") {
        taskWrites++;
        if (state === "ambiguous" && taskWrites === 1)
          return failure(503, "storage_transaction_conflict");
        return json({
          task_ids: ["task_manual"],
          created_count: 1,
          audit_revision: current.revision,
        });
      }
      const h = url.searchParams.get("hypothesis_id");
      return json(
        page(h ? tasks.filter((t) => t.document.hypothesis?.id === h) : tasks),
      );
    }
    if (path.endsWith("/attempts"))
      return json({
        ...page([
          {
            attempt: 1,
            execution_id: "execution_1",
            run_id: "run_1",
            run_deleted: state === "deleted-run",
            state: "settled",
            outcome: "succeeded",
            collection_disposition: "accepted-result",
            created_at: at,
            collected_at: at,
          },
        ]),
        auditRevision: current.revision,
      });
    if (path === `/v1/audits/${RESEARCH_AUDIT}/materials`)
      return json({ ...page(materials), auditRevision: current.revision });
    if (path === `/v1/audits/${RESEARCH_AUDIT}/artifacts`) {
      const key = url.searchParams.get("key") ?? "",
        entry = materials.find(
          (m) =>
            m.key === key &&
            m.material.ref.revision === url.searchParams.get("revision"),
        );
      if (!entry) return failure(404, "not_found");
      return bytes(payloads.get(key)!, entry.material.media_type);
    }
    if (path === `/v1/audits/${RESEARCH_AUDIT}/findings`)
      return json({
        ...page([findingSummary(problem)]),
        auditRevision: current.revision,
        projectRevision: 1,
        asOf: at,
      });
    if (path === `/v1/audits/${RESEARCH_AUDIT}/events`)
      return json({
        ...page([
          {
            auditId: current.auditId,
            sequence: 3,
            kind: "task.settled",
            entityId: "task_0",
            summary: { runId: "run_0", taskCount: 1 },
            createdAt: at,
          },
          {
            auditId: current.auditId,
            sequence: 2,
            kind: "hypothesis.accepted",
            entityId: "hypothesis_3",
            summary: { createdCount: 1 },
            createdAt: at,
          },
          {
            auditId: current.auditId,
            sequence: 1,
            kind: "audit.created",
            entityId: current.auditId,
            summary: {},
            createdAt: at,
          },
        ]),
        throughSequence: 3,
        total: 3,
      });
    if (
      path === `/v1/projects/${RESEARCH_PROJECT}/audit-material-publications`
    ) {
      publicationWrites++;
      if (state === "ambiguous" && publicationWrites === 1)
        return failure(503, "storage_transaction_conflict");
      const body = call.body as Schemas["AuditMaterialPublicationRequest"];
      return json(
        {
          auditId: current.auditId,
          projectId: current.projectId,
          source: body.source,
          artifact: { ...body.target, revision: body.source.revision },
        },
        201,
      );
    }
    return undefined;
  }
  return {
    respond,
    calls,
    tasks,
    hypotheses,
    materials,
    payloads,
    get audit() {
      return current;
    },
    project: owner,
    program,
  };
}
