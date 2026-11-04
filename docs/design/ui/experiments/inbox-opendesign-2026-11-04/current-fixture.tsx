// Real Inbox routes with schema-valid synthetic public API fixtures. No live writes.
import {createRoot} from 'react-dom/client';
import {createMemoryRouter} from 'react-router';
import {Application} from '/src/app/application';
import {applicationRoutes} from '/src/app/router';
import {initializeTheme} from '/src/app/theme';
import {PublicAPI} from '/src/api/client';
import {RunEventsManager} from '/src/events/run-events';
import * as build from '/src/test/builders';
import {runStatus,stageAttempt,workflowResource} from '/src/test/builders-execution';
import {json,page,failure,emptyPageOrNotFound,operationOf} from '/src/test/responses';
import {responseViolation} from '/src/api/contract';
import {ownerListRoutes} from '/src/test/owner-lists';
import type {Audit,ProjectFinding,AuditItem,AuditReport,AuditReviewAction,AuditReviewRequest,AuditWorkspace} from '/src/api/audits';
import type {Project} from '/src/api/projects';
import type {RunStatus,RunSummary} from '/src/api/runs';
import type {WorkflowResource} from '/src/api/workflows';
import type {Routes} from '/src/test/fake-server';
import packageMetadata from '/package.json';
import config from '/@fs//Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/valid/shared-config.json';
import '/src/app/theme.css';import '/src/styles.css';import '/src/app/ui-primitives.css';import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),theme=params.get('theme')??'dark',scenario=params.get('state')??'normal';
localStorage.setItem('contractor.theme',theme);initializeTheme();
const testSession=()=>build.conforming('AuthSession',{principal:{userId:'preview',username:'preview',capabilities:['user','operations']},csrfToken:'a'.repeat(43),idleExpiresAt:'2099-01-01T00:00:00Z',absoluteExpiresAt:'2099-01-02T00:00:00Z'});
const NOW = Date.now();

/** An ISO time `minutes` ago. */
function ago(minutes: number): string {
  return new Date(NOW - minutes * 60_000).toISOString();
}

/** A check last changed 5 minutes ago. */
function audit(
  projectId: string,
  auditId: string,
  state: Audit["state"],
  overrides: Partial<Audit> = {},
): Audit {
  return build.audit(projectId, auditId, state, {
    updatedAt: ago(5),
    ...overrides,
  });
}

/** A possible issue proposed `minutes` ago. */
function finding(
  auditId: string,
  findingId: string,
  title: string,
  minutes: number,
  document: Partial<NonNullable<ProjectFinding["claim"]>> = {},
): ProjectFinding {
  return build.finding(
    auditId,
    findingId,
    { revision: 3, createdAt: ago(minutes), updatedAt: ago(minutes) },
    { title, ...document },
  );
}

/** A pending request made `minutes` ago, about work item item_1 by default. */
function review(
  auditId: string,
  requestId: string,
  kind: AuditReviewRequest["kind"],
  minutes: number,
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return build.review(auditId, requestId, kind, {
    ...(kind === "report-acceptance" ? {} : { subjectId: "item_1" }),
    createdAt: ago(minutes),
    updatedAt: ago(minutes),
    ...overrides,
  });
}

/** Workspace counters of `check`: five items, one done, unless given. */
function workspace(
  check: Audit,
  counts: Parameters<typeof build.workspace>[1] = {},
): AuditWorkspace {
  return build.workspace(check, {
    totalChecks: 5,
    completedChecks: 1,
    unchecked: 4,
    ...counts,
  });
}

/** A Run of the owner's list that changed `minutes` ago. */
function summaryOf(
  runId: string,
  workflow: string,
  state: RunSummary["state"],
  minutes: number,
  projectId?: string,
): RunSummary {
  const terminal = ["succeeded", "failed", "cancelled"].includes(state);
  return build.conforming("RunSummary", {
    runId,
    workflow,
    state,
    deletable: false,
    labels: {},
    ...(projectId === undefined ? {} : { projectId }),
    createdAt: ago(minutes + 5),
    updatedAt: ago(minutes),
    ...(terminal ? { finishedAt: ago(minutes) } : {}),
  });
}

/** The status of a listed Run, as the Server reads it. */
function statusOf(run: RunSummary, extra: Partial<RunStatus> = {}): RunStatus {
  return runStatus(run.runId, {
    workflow: run.workflow,
    state: run.state,
    deletable: false,
    createdAt: run.createdAt,
    updatedAt: run.updatedAt,
    ...(run.finishedAt === undefined ? {} : { finishedAt: run.finishedAt }),
    ...extra,
  });
}

const FAILED_ATTEMPT = stageAttempt("stage_failed_1", {
  stage: "analysis",
  state: "failed",
  termination: {
    outcome: "interrupted",
    code: "worker_gateway_unavailable",
    message: "The model gateway refused the connection.",
    retryable: true,
    phase: "running",
    occurredAt: ago(90),
  },
  createdAt: ago(95),
  updatedAt: ago(90),
});

/** What the fake Server holds; it changes as the page acts on it. */
interface InboxServer {
  projects: Project[];
  audits: Record<string, Audit[]>;
  /** Possible issues in state `proposed`, by check. */
  findings: Record<string, ProjectFinding[]>;
  /** Pending review requests, by check. */
  reviews: Record<string, AuditReviewRequest[]>;
  reports: Record<string, AuditReport>;
  workspaces: Record<string, AuditWorkspace>;
  items: Record<string, AuditItem[]>;
  runs: RunSummary[];
  statuses: Record<string, RunStatus>;
  /** Exact Workflow versions by "name@version". */
  workflows: Record<string, WorkflowResource>;
  paused: boolean;
  /** Answers a request before the fake does, e.g. with a failure. */
  override?: (url: URL, request: Request) => Response | undefined;
  /** Holds a request until the returned promise settles. */
  delay?: (url: URL, request: Request) => Promise<void> | undefined;
}

/** The check with this ID. */
function auditOf(server: InboxServer, auditId: string): Audit {
  const found = Object.values(server.audits)
    .flat()
    .find((entry) => entry.auditId === auditId);
  if (found === undefined) throw new Error(`No check ${auditId}`);
  return found;
}

/** A recorded decision advances its check's revision, as the Server does. */
function advance(server: InboxServer, auditId: string) {
  const entry = auditOf(server, auditId);
  entry.revision += 1;
  entry.updatedAt = ago(0);
}

function emptyServer(): InboxServer {
  return {
    projects: [],
    audits: {},
    findings: {},
    reviews: {},
    reports: {},
    workspaces: {},
    items: {},
    runs: [],
    statuses: {},
    workflows: {},
    paused: false,
  };
}

/** Two projects with something in every section. */
function busyServer(): InboxServer {
  const runs = [
    summaryOf("run_failed", "review@2", "failed", 90),
    summaryOf("run_old_failure", "review@2", "failed", 60 * 24 * 10),
    summaryOf("run_waiting", "triage@3", "waiting", 3),
    summaryOf("run_active", "triage@3", "running", 2),
    summaryOf("run_ok", "generate@1", "succeeded", 30, "project_identity"),
  ];
  const byId = Object.fromEntries(runs.map((run) => [run.runId, run]));
  const trace = audit("project_shop", "audit_trace", "active", {
    revision: 3,
  });
  const wait = audit("project_shop", "audit_wait", "waiting_review");
  const paused = audit("project_shop", "audit_paused", "paused", {
    pausedAt: ago(40),
    stopReason: {
      code: "deadline_exhausted",
      message: "The check used its time limit.",
    },
  });
  const done = audit("project_identity", "audit_done", "completed", {
    finishedAt: ago(60),
  });
  const acceptance = review(
    "audit_wait",
    "review_report",
    "report-acceptance",
    15,
  );
  return {
    ...emptyServer(),
    projects: [
      build.project("project_shop", { name: "crapi-workshop" }),
      build.project("project_identity", { name: "crapi-identity" }),
    ],
    audits: {
      project_shop: [trace, wait, paused],
      project_identity: [
        done,
        audit("project_identity", "audit_failed", "failed", {
          finishedAt: ago(70),
          stopReason: {
            code: "role_execution_not_retryable",
            message: "The trace role failed for good.",
          },
        }),
        audit("project_identity", "audit_finished", "completed", {
          finishedAt: ago(80),
        }),
      ],
    },
    findings: {
      audit_trace: [
        finding(
          "project_shop",
          "finding_idor",
          "Any user can read any report",
          10,
          {
            standard_refs: [
              { scheme: "CWE", version: "4.14", requirement_id: "CWE-639" },
            ],
          },
        ),
        finding(
          "project_shop",
          "finding_sqli",
          "Order search builds raw SQL",
          20,
        ),
      ],
    },
    reviews: {
      audit_wait: [acceptance],
      audit_trace: [
        review("audit_trace", "review_approval", "active-check-approval", 25),
      ],
    },
    reports: {
      audit_wait: build.report("audit_wait", "proposed", {
        review: acceptance,
        summary: "# Report\n\nThree endpoints were traced.",
      }),
      audit_done: build.report("audit_done", "ready", {
        summary: "# Report\n\nAll five endpoints were traced.",
      }),
    },
    workspaces: {
      audit_trace: workspace(trace, { gaps: 2, unchecked: 2 }),
      audit_wait: workspace(wait, { completedChecks: 5, unchecked: 0 }),
      audit_paused: workspace(paused, { completedChecks: 2, unchecked: 3 }),
      audit_done: workspace(done, { completedChecks: 5, unchecked: 0 }),
    },
    items: {
      audit_trace: [
        build.auditItem("item_1", 0, "standard-mapping", {
          itemKey: "WSTG-ATHN-01",
          subjectKey: "WSTG-ATHN-01",
          state: "awaiting_review",
          approvalKind: "active-check-approval",
          createdAt: ago(30),
          updatedAt: ago(25),
        }),
      ],
    },
    runs,
    statuses: {
      run_failed: statusOf(byId.run_failed!, { attempts: [FAILED_ATTEMPT] }),
      run_waiting: statusOf(byId.run_waiting!, {
        recovery: {
          code: "model_unavailable",
          since: ago(20),
          automaticUntil: ago(10),
          requiresRetry: true,
        },
      }),
      run_active: statusOf(byId.run_active!),
      run_ok: statusOf(byId.run_ok!, {
        projectId: "project_identity",
        outputs: {
          report: { namespace: "outputs", name: "report", revision: "r1" },
        },
      }),
    },
    workflows: {
      "generate@1": workflowResource("generate", "1", {
        outputs: {
          report: {
            required: true,
            mediaTypes: ["text/markdown"],
            primary: true,
          },
          log: { required: false, mediaTypes: ["text/plain"] },
        },
      }),
    },
  };
}

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

/** The header of a per-check page: the revision it was read at. */
const CHECK_PAGE = { auditRevision: 1, asOf: ago(1) };

/** A decision the Server records, by action or verdict. */
interface DecisionBody {
  action: AuditReviewAction;
  rationale: string;
}

/** Records the decision on `requestId` and answers as the Server does. */
async function decide(
  server: InboxServer,
  auditId: string,
  requestId: string,
  body: DecisionBody,
): Promise<Response> {
  // The decision leaves the list and advances the check's revision.
  advance(server, auditId);
  const decided = (request: AuditReviewRequest) => {
    const decision = build.decision(request, {
      decisionId: "decision_1",
      actorId: "user_local",
      action: body.action,
      rationale: body.rationale,
      createdAt: ago(0),
    });
    return {
      decision,
      request: {
        ...request,
        state: "decided" as const,
        revision: request.revision + 1,
        decision,
        updatedAt: ago(0),
      },
    };
  };
  const pending = server.reviews[auditId]?.find(
    (candidate) => candidate.requestId === requestId,
  );
  if (pending !== undefined) {
    // An approval, applicability or report acceptance request.
    server.reviews[auditId] = server.reviews[auditId]!.filter(
      (candidate) => candidate !== pending,
    );
    return json({ ...decided(pending), replayed: false });
  }
  return failure(404);
}

/** `found`, or the Server's 404 for a resource that is not there. */
function found<T>(value: T | undefined, headers?: Record<string, string>) {
  return value === undefined ? failure(404) : json(value, 200, headers);
}

/** The Server's answers over `server`, which they read and change. */
function inboxRoutes(server: InboxServer): Routes {
  return {
    ...ownerListRoutes(() => ({
      projects: server.projects,
      audits: Object.values(server.audits).flat(),
      findings: Object.values(server.findings).flat(),
      reviews: Object.values(server.reviews).flat(),
    })),
    "GET /v1/auth/session": () => json(testSession(["user", "operations"])),
    "GET /v1/queue/control": () =>
      server.paused
        ? json({ paused: true, revision: "4", updatedAt: ago(10) }, 200, {
            ETag: '"4"',
          })
        : json({ paused: false, revision: "0" }, 200, { ETag: '"0"' }),
    "GET /v1/projects": () => json(page(server.projects)),
    "GET /v1/projects/{projectId}/audits": ({ params, url }) => {
      const state = url.searchParams.get("state");
      const audits = server.audits[params.projectId!] ?? [];
      return json(
        page(state === null ? audits : audits.filter((a) => a.state === state)),
      );
    },
    "GET /v1/audits/{auditId}": ({ params }) => {
      const check = Object.values(server.audits)
        .flat()
        .find((entry) => entry.auditId === params.auditId);
      return found(check, { ETag: `"${check?.revision}"` });
    },
    "GET /v1/audits/{auditId}/findings": ({ params, url }) => {
      const items =
        url.searchParams.get("state") === "proposed"
          ? (server.findings[params.auditId!] ?? [])
          : [];
      return json({
        ...CHECK_PAGE,
        total: items.length,
        ...page(items.map(build.findingSummary)),
      });
    },
    "GET /v1/audits/{auditId}/findings/{findingId}": ({ params }) => {
      const subject = server.findings[params.auditId!]?.find(
        (candidate) => candidate.findingId === params.findingId,
      );
      return found(subject, { ETag: `"${subject?.revision}"` });
    },

    "GET /v1/projects/{projectId}/findings/{findingId}": ({ params }) => {
      const value = Object.values(server.findings)
        .flat()
        .find(
          (f) =>
            f.projectId === params.projectId &&
            f.findingId === params.findingId,
        );
      return found(value, { ETag: `"${value?.revision}"` });
    },
    "PATCH /v1/projects/{projectId}/findings/{findingId}": async ({
      params,
      request,
    }) => {
      const value = Object.values(server.findings)
        .flat()
        .find(
          (f) =>
            f.projectId === params.projectId &&
            f.findingId === params.findingId,
        );
      if (value === undefined) return failure(404);
      if (request.headers.get("If-Match") !== `"${value.revision}"`)
        return failure(412, "precondition_failed", "Changed first");
      const body =
        (await request.json()) as import("../../api/audits").EditProjectFindingRequest;
      const states = {
        true_positive: "confirmed",
        false_positive: "rejected",
        duplicate: "duplicate",
        reopen: "proposed",
        needs_evidence: "needs-evidence",
      } as const;
      const next: ProjectFinding = {
        ...value,
        state: states[body.verdict],
        analystVerdict: body.verdict,
        analystSeverity:
          body.verdict === "true_positive" ? body.severity : null,
        duplicateTargetId:
          body.verdict === "duplicate" ? body.duplicateTargetId : null,
        rationale: body.rationale,
        actorId: "user_local",
        assessedAt: ago(0),
        revision: value.revision + 1,
      };
      for (const [key, values] of Object.entries(server.findings))
        server.findings[key] = values.map((f) =>
          f.findingId === next.findingId ? next : f,
        );
      return json({ finding: next, replayed: false }, 200, {
        ETag: `"${next.revision}"`,
      });
    },
    "GET /v1/audits/{auditId}/reviews": ({ params, url }) => {
      const items = (server.reviews[params.auditId!] ?? []).filter(
        (candidate) =>
          url.searchParams.get("state") === null ||
          candidate.state === url.searchParams.get("state"),
      );
      return json({ ...CHECK_PAGE, total: items.length, ...page(items) });
    },
    "POST /v1/audits/{auditId}/reviews/{requestId}/decisions": async ({
      params,
      request,
    }) =>
      decide(
        server,
        params.auditId!,
        params.requestId!,
        (await request.json()) as DecisionBody,
      ),
    "GET /v1/audits/{auditId}/report": ({ params }) =>
      found(server.reports[params.auditId!]),
    "GET /v1/audits/{auditId}/workspace": ({ params }) =>
      found(server.workspaces[params.auditId!]),
    "GET /v1/audits/{auditId}/items": ({ params, url }) => {
      const state = url.searchParams.get("state");
      return json(
        page(
          (server.items[params.auditId!] ?? []).filter(
            (item) => state === null || item.state === state,
          ),
        ),
      );
    },
    "GET /v1/runs": ({ url }) => {
      const state = url.searchParams.get("state");
      const lifecycle = url.searchParams.get("lifecycle");
      return json(
        page(
          server.runs.filter(
            (run) =>
              (state === null || run.state === state) &&
              (lifecycle === null ||
                (lifecycle === "active") !== TERMINAL.has(run.state)),
          ),
        ),
      );
    },
    "GET /v1/runs/{runId}": ({ params }) =>
      found(server.statuses[params.runId!]),
    "POST /v1/runs/{runId}/retry-gateway": ({ params }) => {
      const runId = params.runId!;
      const status = server.statuses[runId];
      if (status !== undefined) {
        delete status.recovery;
        status.state = "running";
      }
      server.runs = server.runs.map((run) =>
        run.runId === runId ? { ...run, state: "running" } : run,
      );
      return json({ runId }, 202);
    },
    "GET /v1/workflows/{name}/versions/{version}": ({ params }) =>
      found(server.workflows[`${params.name}@${params.version}`]),
  };
}


const server=scenario==='empty'?emptyServer():busyServer();
if(params.get('view')==='running'&&scenario==='normal'){const active=server.audits.project_shop.find(a=>a.auditId==='audit_trace');server.workspaces.audit_trace=workspace(active,{gaps:0,unchecked:4});}
const routes=inboxRoutes(server);
window.__inboxRequests=[];window.__inboxViolations=[];
const api=new PublicAPI({uiVersion:packageMetadata.version,supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url),path=url.pathname;
 window.__inboxRequests.push({method:request.method,path});
 if(request.method!=='GET')return failure(403,'forbidden','Design preview: decisions and retries are disabled; no live mutation was sent.');
 let response;
 if(path==='/v1/config')response=json(config);
 else if((scenario==='error'&&['/v1/audits','/v1/findings','/v1/reviews','/v1/runs'].includes(path))||(scenario==='partial'&&path==='/v1/findings'))response=failure(503,'storage_unavailable','Synthetic Inbox read failure');
 else {
  for(const [pattern,handler] of Object.entries(routes)){
   const [method,template]=pattern.split(' ');if(method!==request.method)continue;
   const names=[];const regex=new RegExp('^'+template.replace(/\{([^}]+)\}/g,(_,name)=>{names.push(name);return '([^/]+)'})+'$'),match=path.match(regex);if(!match)continue;
   response=await handler({request,url,params:Object.fromEntries(names.map((name,i)=>[name,decodeURIComponent(match[i+1])]))});break;
  }
  response??=emptyPageOrNotFound(request,url);
 }
 const operation=operationOf(request.method,path);
 if(operation&&response.headers.get('Content-Type')?.includes('json')){
  const violation=responseViolation(operation.operation,response.status,await response.clone().json());if(violation)window.__inboxViolations.push({path,violation});
 }
 return response;
});
class DemoSocket{protocol='contractor.events.v1';readyState=0;onopen=null;onmessage=null;onerror=null;onclose=null;constructor(){queueMicrotask(()=>{this.readyState=1;this.onopen?.(new Event('open'))})}send(value){const f=JSON.parse(value);if(f.type==='subscribe')queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({version:'contractor.events.v1',type:'subscribed',subscriptionId:f.subscriptionId,stream:f.stream,cursor:f.after})}))}close(code=1000){this.readyState=3;this.onclose?.({code})}}
const views={overview:'',issue:'issue:project_shop:finding_idor',review:'review:audit_trace:review_approval',blocked:'run:run_waiting',ready:'run:run_ok',running:'check:audit_trace'};
const view=params.get('view')??'overview',entry=views[view]?'/'+ '?item='+encodeURIComponent(views[view]):'/';
const events=new RunEventsManager(location.origin,{WebSocketImplementation:DemoSocket});
if(params.get('version')!=='current'){
 document.body.classList.add('inbox-redesign');
 for(const name of ['opendesign.css','layout-fixes.css']){const style=document.createElement('link');style.rel='stylesheet';style.href='/test-results/inbox-design/'+name+'?rev=REVISION';document.head.append(style)}
}
createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} runEvents={events} router={createMemoryRouter(applicationRoutes(),{initialEntries:[entry]})}/>);
