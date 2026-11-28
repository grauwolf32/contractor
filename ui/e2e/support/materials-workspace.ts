// Synthetic review data and stateful API adapter. This module is test-only.
import examples from "../../src/test/fixtures/materials-workspace.json" with { type: "json" };
import engagement from "../../src/test/fixtures/engagement-preview.json" with { type: "json" };
import sharedConfig from "../../../api/testdata/v1alpha1/valid/shared-config.json" with { type: "json" };
import {
  artifactMetadata,
  artifactWrite,
  conforming,
} from "../../src/test/builders";
import {
  failure,
  json,
  bytes,
  operationOf,
  unavailableCostForURL,
} from "../../src/test/responses";
import { responseViolation } from "../../src/api/contract";
import {
  profileFixture,
  projectFixture,
  draftAudit,
  startResponse,
} from "../../src/routes/checks/start/test-fixtures";
import type { ArtifactMetadata } from "../../src/api/artifacts";

const now = "2026-11-28T09:15:00Z";
export const materialsProject = projectFixture({
  name: "Invoice Lab",
  description:
    "Security review of the invoice API and its supporting services.",
});
export const materialsProfiles = [
  profileFixture("owasp-top10-2025-source-risk"),
  profileFixture("openapi-operation-trace", {
    inputs: {
      source: { required: true, media_types: ["application/zip"] },
      openapi: {
        required: true,
        media_types: ["application/yaml", "application/json"],
      },
    },
  }),
  profileFixture("owasp-wstg-4-2-active-http", {
    inputs: {
      openapi: {
        required: true,
        media_types: ["application/yaml", "application/json"],
      },
      engagement: {
        required: true,
        media_types: ["application/vnd.contractor.engagement+json"],
      },
    },
    policy: { active_operations: "allowed" },
  }),
];
export const materialsSession = conforming("AuthSession", {
  principal: {
    userId: "materials-review",
    username: "review",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-01-01T00:00:00Z",
  absoluteExpiresAt: "2099-01-02T00:00:00Z",
});
export function createMaterialsReviewAPI() {
  const files = examples.map((example) => {
    const mediaType =
      example.kind === "source"
        ? "application/zip"
        : example.kind === "api"
          ? "application/yaml"
          : example.kind === "engagement"
            ? "application/vnd.contractor.engagement+json"
            : example.kind === "docs"
              ? "text/markdown"
              : example.kind === "diffs"
                ? "text/x-diff"
                : example.kind === "architecture"
                  ? "text/vnd.likec4"
                  : "application/json";
    const content =
      example.kind === "engagement"
        ? JSON.stringify(engagement)
        : example.content;
    const metadata = artifactMetadata(example.namespace, example.name, {
      artifact: {
        namespace: example.namespace,
        name: example.name,
        revision: example.revision,
      },
      mediaType,
      size: new TextEncoder().encode(content).length,
      createdAt: now,
      ...(example.commit
        ? {
            gitSource: {
              repositoryUrl: "https://example.test/invoice.git",
              requestedRef: "main",
              resolvedCommit: example.commit.padEnd(40, "0"),
              importedAt: now,
            },
          }
        : {}),
    });
    return { metadata, content, paths: example.files ?? [] };
  });
  let draft: ReturnType<typeof draftAudit> | undefined;
  async function fetchReview(input: RequestInfo | URL): Promise<Response> {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
    const pathname = url.pathname;
    const tag = (revision: string | number) => ({ ETag: `"${revision}"` });
    let response: Response;
    if (pathname === "/v1/config") response = json(sharedConfig);
    else if (pathname === "/v1/auth/session") response = json(materialsSession);
    else if (pathname === "/v1/projects")
      response = json({ items: [materialsProject], page: { hasMore: false } });
    else if (pathname === "/v1/projects/project_shop")
      response = json(materialsProject, 200, tag(materialsProject.revision));
    else if (pathname === "/v1/audit-profiles")
      response = json({ items: materialsProfiles, page: { hasMore: false } });
    else if (pathname.startsWith("/v1/audit-profiles/")) {
      const found = materialsProfiles.find(
        (profile) =>
          pathname ===
          `/v1/audit-profiles/${profile.ref.name}/versions/${profile.ref.version}`,
      );
      response =
        found === undefined
          ? failure(404)
          : json(found, 200, tag(found.ref.digest));
    } else if (pathname === "/v1/projects/project_shop/artifacts") {
      response = json({
        items: files
          .filter(
            (file) =>
              !url.searchParams.has("namespace") ||
              file.metadata.artifact.namespace ===
                url.searchParams.get("namespace"),
          )
          .map((file) => file.metadata),
        page: { hasMore: false },
      });
    } else if (pathname.startsWith("/v1/projects/project_shop/artifacts/")) {
      const [namespace, name, action, fileAction] = pathname
        .slice("/v1/projects/project_shop/artifacts/".length)
        .split("/")
        .map(decodeURIComponent);
      const found = files.find(
        (file) =>
          file.metadata.artifact.namespace === namespace &&
          file.metadata.artifact.name === name,
      );
      if (request.method === "PUT" && namespace && name) {
        if (found !== undefined) response = failure(412, "precondition_failed");
        else {
          const content = await request.text();
          const metadata = artifactMetadata(namespace, name, {
            artifact: { namespace, name, revision: `added-${files.length}` },
            mediaType: request.headers.get("content-type") ?? "text/plain",
            size: new TextEncoder().encode(content).length,
            createdAt: now,
          });
          files.unshift({ metadata, content, paths: ["uploaded-material"] });
          response = json(
            artifactWrite(metadata),
            201,
            tag(metadata.artifact.revision),
          );
        }
      } else if (
        found === undefined ||
        (url.searchParams.has("revision") &&
          found.metadata.artifact.revision !== url.searchParams.get("revision"))
      )
        response = failure(404);
      else if (action === "metadata")
        response = json(
          found.metadata,
          200,
          tag(found.metadata.artifact.revision),
        );
      else if (action === "versions")
        response = json({ items: [found.metadata], page: { hasMore: false } });
      else if (action === "lineage")
        response = json({ items: [], page: { hasMore: false } });
      else if (action === "archive") {
        const ref = found.metadata.artifact;
        if (fileAction === "file")
          response = json(
            {
              artifact: ref,
              path: url.searchParams.get("path"),
              size: new TextEncoder().encode(found.content).length,
              text: found.content,
            },
            200,
            tag(ref.revision),
          );
        else {
          const paths = new Set(
            found.paths.flatMap((path) =>
              path
                .split("/")
                .map((_, index, parts) => parts.slice(0, index + 1).join("/")),
            ),
          );
          response = json(
            {
              artifact: ref,
              entries: [...paths].map((path) => ({
                path,
                kind: found.paths.includes(path) ? "file" : "directory",
                size: found.paths.includes(path)
                  ? new TextEncoder().encode(found.content).length
                  : 0,
                previewable: found.paths.includes(path),
              })),
            },
            200,
            tag(ref.revision),
          );
        }
      } else
        response = bytes(
          found.content,
          found.metadata.mediaType,
          tag(found.metadata.artifact.revision),
        );
    } else if (
      pathname === "/v1/projects/project_shop/audits" &&
      request.method === "POST"
    ) {
      const body = (await request.json()) as { profile: { name: string } };
      draft = draftAudit(
        materialsProfiles.find(
          (profile) => profile.ref.name === body.profile.name,
        )!,
      );
      response = json(draft, 201, tag(draft.revision));
    } else if (pathname.endsWith("/start") && draft !== undefined)
      response = json(startResponse(draft));
    else {
      const cost = unavailableCostForURL(request.method, url);
      const operation = operationOf(request.method, pathname)?.operation;
      const empty = { items: [], page: { hasMore: false } };
      response = cost
        ? json(cost)
        : operation !== undefined &&
            responseViolation(operation, 200, empty) === undefined
          ? json(empty)
          : failure(404);
    }
    return response;
  }
  return {
    fetch: fetchReview,
    items: () => files.map((file) => file.metadata) as ArtifactMetadata[],
  };
}
