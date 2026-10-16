// A fake Server and router around the Start page, for its tests.
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { Audit, AuditProfile } from "../../../api/audits";
import type { Project } from "../../../api/projects";
import { fakeServer, type FakeCall } from "../../../test/fake-server";
import { failure, json, page } from "../../../test/responses";
import { renderWithAPI } from "../../../test/render";
import { StartCheckRoute } from "../start";
import { draftAudit, projectFixture, startResponse } from "./test-fixtures";

type Answer = (request: Request, url: URL) => Response | Promise<Response>;

export interface FakeStartServer {
  /** Projects listed by the picker. */
  projects?: Project[];
  /** The project read by ID; null answers 404. */
  project?: Project | null;
  profiles?: AuditProfile[];
  /** GET /v1/audit-profiles; one page of `profiles` otherwise. */
  profileList?: Answer;
  /** Detail responses by "name@version"; the list item otherwise. */
  details?: Record<string, AuditProfile>;
  materials?: ArtifactMetadata[] | Answer;
  /** POST /v1/projects/{id}/audits; a draft of the request's profile otherwise. */
  create?: Answer;
  /** POST /v1/audits/{id}/start; the started draft otherwise. */
  start?: Answer;
  /**
   * GET /v1/audits/{id}; otherwise the check as the default create and start
   * answers left it (a draft at revision 1 when they were replaced).
   */
  audit?: Answer;
  /** A page whose code loads only once `until` settles: a slow next page. */
  slowPage?: { path: string; until: Promise<void> };
}

export function renderStart(path: string, server: FakeStartServer = {}) {
  const project =
    server.project === undefined ? projectFixture() : server.project;
  const profiles = server.profiles ?? [];
  // Checks as the default create and start answers left them.
  const audits = new Map<string, Audit>();
  const answer =
    (custom: Answer | undefined, fallback: (call: FakeCall) => Response) =>
    (call: FakeCall) =>
      custom === undefined ? fallback(call) : custom(call.request, call.url);

  const fake = fakeServer({
    "GET /v1/projects": () => json(page(server.projects ?? [])),
    "GET /v1/projects/{projectId}": ({ params }) =>
      project === null || project.projectId !== params.projectId
        ? failure(404, "not_found", "Project not found")
        : json(project, 200, { ETag: `"${project.revision}"` }),
    "GET /v1/audit-profiles": answer(server.profileList, () =>
      json(page(profiles)),
    ),
    "GET /v1/audit-profiles/{name}/versions/{version}": ({ params }) => {
      const key = `${params.name}@${params.version}`;
      const found =
        server.details?.[key] ??
        profiles.find(
          (profile) => `${profile.ref.name}@${profile.ref.version}` === key,
        );
      return found === undefined
        ? failure(404, "not_found", "Check type not found")
        : json(found, 200, { ETag: `"${found.ref.digest}"` });
    },
    "GET /v1/projects/{projectId}/artifacts": ({ request, url }) => {
      const materials = server.materials ?? [];
      return typeof materials === "function"
        ? materials(request, url)
        : json(page(materials));
    },
    "GET /v1/projects/{projectId}/audits": () => json(page([])),
    "POST /v1/projects/{projectId}/audits": async (call) => {
      if (server.create !== undefined)
        return server.create(call.request, call.url);
      const body = (await call.request.clone().json()) as {
        profile: { name: string; version: string };
      };
      const profile = profiles.find(
        (candidate) =>
          candidate.ref.name === body.profile.name &&
          candidate.ref.version === body.profile.version,
      );
      if (profile === undefined)
        return failure(404, "not_found", "Check type not found");
      const draft = draftAudit(profile);
      audits.set(draft.auditId, draft);
      return json(draft, 201, { ETag: '"1"' });
    },
    "POST /v1/audits/{auditId}/start": answer(server.start, ({ params }) => {
      const auditId = params.auditId ?? "";
      const draft =
        audits.get(auditId) ?? draftAudit(profiles[0]!, { auditId });
      const started = startResponse(draft);
      audits.set(auditId, started.audit);
      return json(started, 200, { ETag: `"${started.audit.revision}"` });
    }),
    "GET /v1/audits/{auditId}": answer(server.audit, ({ params }) => {
      const auditId = params.auditId ?? "";
      const audit =
        audits.get(auditId) ?? draftAudit(profiles[0]!, { auditId });
      return json(audit, 200, { ETag: `"${audit.revision}"` });
    }),
  });

  return renderWithAPI(null, {
    path,
    server: fake,
    routes: [
      { path: "/checks/new", element: <StartCheckRoute /> },
      ...(server.slowPage === undefined
        ? []
        : [
            {
              path: server.slowPage.path,
              lazy: async () => {
                await server.slowPage?.until;
                return { element: <p>The slow page</p> };
              },
            },
          ]),
      // Elsewhere: tests read router.state.location.
      { path: "*", element: <p>Another page</p> },
    ],
  });
}
