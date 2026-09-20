import draftExample from "../../../api/testdata/evals/valid/draft-experiment.json" with { type: "json" };
import experimentExample from "../../../api/testdata/evals/valid/experiment.json" with { type: "json" };
import datasetExample from "../../../api/testdata/evals/valid/dataset.json" with { type: "json" };
import datasetMetadata from "../../../api/testdata/evals/valid/dataset-view.json" with { type: "json" };
import pairsExample from "../../../api/testdata/evals/valid/pair-page.json" with { type: "json" };
import membersExample from "../../../api/testdata/evals/valid/member-page.json" with { type: "json" };
import reviewExample from "../../../api/testdata/evals/valid/review.json" with { type: "json" };
import resultExample from "../../../api/testdata/evals/valid/result.json" with { type: "json" };
import quality from "../../../api/testdata/evals/valid/quality-chart.json" with { type: "json" };
import tokens from "../../../api/testdata/evals/valid/tokens-chart.json" with { type: "json" };
import duration from "../../../api/testdata/evals/valid/duration-chart.json" with { type: "json" };
import progress from "../../../api/testdata/evals/valid/progress-chart.json" with { type: "json" };
import differences from "../../../api/testdata/evals/valid/deltas-chart.json" with { type: "json" };
import type { components } from "../api/generated/public";
import type {
  EvalCommand,
  EvalDatasetInput,
  EvalDraft,
  EvalExperiment,
} from "../api/evals";

export const EVAL_API_VERSION = "contractor.public.v1";
export const EVAL_FIXTURE_ORIGIN = "http://127.0.0.1:8080";
export const EVAL_FIXTURE_DIGEST = experimentExample.planSha256;
export const EVAL_FIXTURE_SESSION = {
  principal: {
    userId: "user_eval",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-01-01T00:00:00Z",
  absoluteExpiresAt: "2099-01-02T00:00:00Z",
};
export interface FixtureResponse {
  body: unknown;
  status: number;
  headers: Record<string, string>;
}
const page = { hasMore: false, nextCursor: null };

export function createEvalFixture(
  options: { prepared?: boolean; external?: boolean; audit?: boolean } = {},
) {
  const initial = structuredClone(
    options.prepared ? experimentExample : draftExample,
  ) as EvalExperiment;
  initial.revision = 1;
  if (options.prepared) {
    initial.state = "ready";
    initial.allowedCommands = ["start", "duplicate"];
  }
  if (options.external) {
    initial.controlMode = "external";
    initial.state = "running";
    initial.allowedCommands = ["finalize"];
  }
  if (options.audit) {
    initial.executionKind = "audit";
    for (const variant of initial.draft?.variants ??
      initial.setup?.variants ??
      []) {
      variant.kind = "audit";
      variant.selector = variant.selector.replace("trace-", "audit-");
    }
  }
  const setup = initial.draft ?? initial.setup;
  if (setup)
    for (const check of setup.checks)
      check.implementationSha256 = EVAL_FIXTURE_DIGEST;
  const fixture = {
    experiment: initial,
    requests: [] as {
      method: string;
      path: string;
      body: unknown;
      key: string | null;
      etag: string | null;
    }[],
    lostCommand: false,
    lostAssessment: false,
    staleSelection: false,
    stalePairs: false,
    staleReview: false,
    secondPairPage: false,
    dataset: structuredClone(datasetExample) as EvalDatasetInput,
  };
  const receipts = new Map<
    string,
    { body: string; etag: string | null; response: FixtureResponse }
  >();
  const commands = new Map<string, unknown>();
  function response(
    body: unknown,
    status = 200,
    revision?: number | string,
  ): FixtureResponse {
    return {
      body,
      status,
      headers: {
        "Content-Type": "application/json",
        "X-Contractor-API-Version": EVAL_API_VERSION,
        ...(revision ? { ETag: `"${revision}"` } : {}),
      },
    };
  }
  function error(code: string, status = 409) {
    return response(
      { code, message: code, details: { kind: "eval", recovery: "reload" } },
      status,
    );
  }
  const project = {
    projectId: "evaluation-1",
    kind: "evaluation",
    name: "Evaluation workspace",
    description: "Fixture workspace",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-09-20T10:00:00Z",
    updatedAt: "2026-09-20T10:00:00Z",
  };
  async function handle(request: Request): Promise<FixtureResponse> {
    const url = new URL(request.url),
      path = url.pathname,
      method = request.method;
    const raw = method === "GET" ? "" : await request.text();
    const body = raw ? (JSON.parse(raw) as Record<string, unknown>) : undefined;
    const key = request.headers.get("Idempotency-Key"),
      etag = request.headers.get("If-Match");
    fixture.requests.push({ method, path, body, key, etag });
    const receiptKey = `${method}:${path}:${key}`;
    if (method !== "GET") {
      if (
        request.headers.get("X-CSRF-Token") !== EVAL_FIXTURE_SESSION.csrfToken
      )
        return error("csrf_required", 403);
      if (!key) return error("eval_invalid", 400);
      const saved = receipts.get(receiptKey);
      if (saved)
        return saved.body === raw && saved.etag === etag
          ? saved.response
          : error("eval_idempotency_conflict");
    }
    const e = fixture.experiment;
    let result: FixtureResponse;
    if (path === "/v1/auth/session") return response(EVAL_FIXTURE_SESSION);
    if (path === "/v1/projects") return response({ items: [project], page });
    if (path === "/v1/eval-capabilities")
      return response({
        controlModes: ["server", "external"],
        executionKinds: ["workflow", "audit"],
        checks: [
          "required-artifact@1",
          "human-review@1",
          "media-type@1",
          "json-schema@1",
        ].map((evaluator) => ({
          evaluator,
          implementationSha256: EVAL_FIXTURE_DIGEST,
          available: true,
          reason: null,
        })),
        schemas: ["json@1"],
        importVersions: ["dataset@1"],
        bindings: ["workflow", "audit"].flatMap((kind) =>
          ["a", "b"].flatMap((arm) =>
            ["1", "2"].map((version) => ({
              kind,
              selector: `${kind === "audit" ? "audit" : "trace"}-${arm}@${version}`,
              available: true,
              reason: null,
            })),
          ),
        ),
        page,
      });
    if (path.includes("/workflows/") && path.includes("/versions/"))
      return response({
        ref: { name: path.split("/")[3], version: path.split("/")[5] },
        entryStage: "check",
        parameters: {},
        inputs: {},
        outputs: {
          report: { required: true, mediaTypes: ["application/json"] },
        },
        stages: {},
      });
    if (path.includes("/audit-profiles/"))
      return response(
        {
          ref: {
            name: path.split("/")[3],
            version: path.split("/")[5],
            digest: EVAL_FIXTURE_DIGEST,
          },
          mode: "custom-checklist",
          standards: [],
          inputs: {},
          inventory: {
            mode: "tasks",
            source: "manifest",
            manifestInput: "tasks",
          },
          execution: {},
          interaction: {},
          serverCompatible: true,
          requiresInputValidation: false,
          compatibilityReasons: [],
        },
        200,
        EVAL_FIXTURE_DIGEST,
      );
    if (
      path.includes("/configurations/") ||
      path === "/v1/operations/credentials" ||
      path.endsWith("/artifacts")
    )
      return response({ items: [], page });
    if (path.endsWith("/eval-datasets") && method === "GET")
      return response({
        items: [
          {
            ...datasetMetadata,
            datasetId: fixture.dataset.datasetId,
            name: fixture.dataset.name,
          },
        ],
        page,
      });
    if (path.endsWith("/cases"))
      return response({ items: fixture.dataset.cases, page });
    if (path.endsWith("/eval-datasets") && method === "POST") {
      fixture.dataset = body as unknown as EvalDatasetInput;
      result = response(
        {
          ...datasetMetadata,
          datasetId: fixture.dataset.datasetId,
          name: fixture.dataset.name,
          caseCount: fixture.dataset.cases.length,
        },
        201,
      );
    } else if (path === "/v1/eval-experiments") {
      return response({
        items: [
          {
            ...e,
            draft: undefined,
            setup: undefined,
            variants: (e.draft?.variants ?? e.setup?.variants)?.map((v) => ({
              id: v.id,
              selector: v.selector,
            })),
            caseCount: 2,
            repetitions: 2,
          },
        ],
        page,
      });
    } else if (path.endsWith("/eval-experiments") && method === "POST") {
      e.name = body!.name as string;
      e.draft = body!.draft as EvalDraft;
      e.executionKind = e.draft.variants[0]!.kind;
      e.expectedMembers = e.draft.caseIds.length * 2 * e.draft.repetitions;
      result = response(
        { experimentId: e.experimentId, revision: e.revision, state: e.state },
        201,
        e.revision,
      );
    } else if (
      path === `/v1/eval-experiments/${e.experimentId}` &&
      method === "GET"
    )
      return response(e, 200, e.revision);
    else if (
      path === `/v1/eval-experiments/${e.experimentId}` &&
      method === "PATCH"
    ) {
      if (etag !== `"${e.revision}"`)
        return error("eval_revision_mismatch", 412);
      e.name = body!.name as string;
      e.draft = body!.draft as EvalDraft;
      e.revision++;
      result = response(
        { experimentId: e.experimentId, revision: e.revision, state: e.state },
        201,
        e.revision,
      );
    } else if (path.endsWith("/commands") && method === "POST") {
      if (e.controlMode !== "server") return error("eval_external_control");
      if (etag !== `"${e.revision}"`)
        return error("eval_revision_mismatch", 412);
      const command = body as unknown as EvalCommand;
      e.revision++;
      if (command.kind === "prepare") {
        e.setup = structuredClone(e.draft!);
        delete e.draft;
        e.state = "ready";
        e.planSha256 = EVAL_FIXTURE_DIGEST;
        e.allowedCommands = ["start", "duplicate"];
        e.viewSnapshot = "view-7";
        e.readiness = {
          pins: [
            {
              dimension: "source",
              requiredEqual: true,
              baselineOrigin: "observed",
              candidateOrigin: "observed",
              status: "equal",
            },
          ],
          arms: ["a", "b"].map((variantId) => ({
            variantId,
            expected: e.expectedMembers / 2,
            eligible: e.expectedMembers / 2,
            unsupported: 0,
            blocked: 0,
          })),
        };
      } else if (command.kind === "start" || command.kind === "resume") {
        e.state = "running";
        e.allowedCommands = ["pause", "cancel", "duplicate"];
      } else if (command.kind === "pause") {
        e.state = "paused";
        e.allowedCommands = ["resume", "cancel", "duplicate"];
      } else if (command.kind === "cancel") {
        e.state = "cancelled";
        e.allowedCommands = ["duplicate"];
      } else if (command.kind === "duplicate") {
        e.experimentId = "experiment-duplicate";
        e.state = "draft";
        e.draft = structuredClone(draftExample.draft) as EvalDraft;
        e.planSha256 = null;
        e.allowedCommands = ["prepare"];
        result = response(
          {
            experimentId: e.experimentId,
            revision: e.revision,
            state: e.state,
          },
          201,
        );
        receipts.set(receiptKey, { body: raw, etag, response: result });
        return result;
      }
      const commandId = `command-${commands.size + 1}`;
      const receipt = {
        commandId,
        experimentId: e.experimentId,
        kind: command.kind,
        state: "completed",
        experimentRevision: e.revision,
        planSha256: e.planSha256,
        diagnostics: [],
      };
      commands.set(commandId, receipt);
      result = response(receipt, 202, e.revision);
    } else if (path.includes("/commands/"))
      return response(commands.get(path.split("/").at(-1)!)!);
    else if (path.endsWith("/members")) return response(membersExample);
    else if (path.endsWith("/pairs")) {
      if (fixture.stalePairs) return error("eval_view_changed");
      return response({
        ...pairsExample,
        filteredCount: fixture.secondPairPage
          ? 100
          : pairsExample.filteredCount,
        page:
          fixture.secondPairPage && !url.searchParams.has("cursor")
            ? { hasMore: true, nextCursor: "pair-page-2" }
            : page,
      });
    } else if (path.includes("/pairs/"))
      return response({
        viewSnapshot: "view-7",
        freshness: "current",
        experimentSummary: experimentExample.summary,
        pair: pairsExample.items[0],
        records: [
          {
            memberId: resultExample.memberId,
            kind: "result",
            recordSha256: EVAL_FIXTURE_DIGEST,
            actorId: "system:eval-collector",
            previousRecordSha256: null,
            createdAt: "2026-09-20T10:00:00Z",
            document: resultExample,
          },
        ],
      });
    else if (path.endsWith("/review"))
      return response({
        ...reviewExample,
        resultSha256: fixture.staleReview
          ? "sha256:" + "f".repeat(64)
          : reviewExample.resultSha256,
        result: resultExample,
        policy: experimentExample.setup.checks.map((check) => ({
          ...check,
          implementationSha256: EVAL_FIXTURE_DIGEST,
        })),
        revision: e.revision,
      });
    else if (path.endsWith("/assessments")) {
      result = response(
        {
          memberId: reviewExample.memberId,
          recordSha256: "sha256:" + "e".repeat(64),
          previousRecordSha256: null,
        },
        201,
      );
    } else if (path.endsWith("/selections")) {
      if (fixture.staleSelection || etag !== `"${e.revision}"`)
        return error("eval_revision_mismatch", 412);
      e.revision++;
      result = response(
        { revision: e.revision, viewSnapshot: "view-8" },
        201,
        e.revision,
      );
    } else if (path.endsWith("/executions"))
      return response({
        inventoryRevision: 1,
        inventoryComplete: true,
        items: [
          {
            execution: {
              kind: options.audit ? "audit" : "run",
              id: options.audit ? "audit-1" : "run-1",
            },
            parent: null,
            role: null,
            round: null,
            state: "succeeded",
            available: true,
            projectId: "member-project",
          },
        ],
        gaps: [],
        page,
      });
    else if (path.includes("/charts/")) {
      const charts: Record<string, unknown> = {
        quality,
        tokens,
        duration,
        progress,
        "pair-deltas": differences,
      };
      return response(charts[path.split("/").at(-1)!]);
    } else if (path.endsWith("/report"))
      return response({
        schemaVersion: "contractor.eval-report/v1",
        experimentId: e.experimentId,
        planSha256: e.planSha256,
        viewSnapshot: "view-7",
        summary: experimentExample.summary,
        sources: [],
        pairCount: 4,
      });
    else throw new Error(`Unexpected fixture request ${method} ${path}`);
    receipts.set(receiptKey, {
      body: raw,
      etag,
      response: structuredClone(result),
    });
    if (path.endsWith("/commands") && fixture.lostCommand) {
      fixture.lostCommand = false;
      throw new TypeError("Simulated lost command response");
    }
    if (path.endsWith("/assessments") && fixture.lostAssessment) {
      fixture.lostAssessment = false;
      throw new TypeError("Simulated lost assessment response");
    }
    return result;
  }
  return {
    state: fixture,
    handle,
    async fetch(request: Request) {
      const result = await handle(request);
      return new Response(JSON.stringify(result.body), {
        status: result.status,
        headers: result.headers,
      });
    },
  };
}

export type FixtureExperiment = components["schemas"]["EvalExperiment"];
