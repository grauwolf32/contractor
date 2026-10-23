import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import ts from "typescript";
import nameCases from "../../../api/testdata/v1alpha1/artifact-name-cases.json";
import reservationCases from "../../../api/testdata/v1alpha1/memory-reservation-cases.json";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import {
  ARTIFACT_NAME_PATTERN,
  canPreviewArtifact,
  downloadArtifact,
  FINDING_PROPOSAL_NAMESPACE,
  listArtifacts,
  MAXIMUM_PREVIEW_BYTES,
  previewArtifact,
  writeArtifact,
  type ArtifactMetadata,
} from "./artifacts";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const metadata: ArtifactMetadata = {
  artifact: {
    namespace: "projects",
    name: "architecture",
    revision: "revision-2",
  },
  mediaType: "text/vnd.likec4",
  size: 17,
  current: true,
  frozen: false,
  createdAt: "2026-08-31T12:00:00Z",
};

function response(body: BodyInit | null, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

/** The filename downloadArtifact suggests for `metadata` in `mediaType`. */
async function downloadedFilename(mediaType: string): Promise<string> {
  const content = "x".repeat(metadata.size);
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async () =>
      response(content, {
        headers: {
          "content-type": mediaType,
          "content-length": String(content.length),
        },
      }),
    ),
  );
  return (await downloadArtifact(api, { ...metadata, mediaType })).filename;
}

describe("Artifact API", () => {
  it("uses generated list parameters and maps the authoritative page", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response(
          JSON.stringify({
            items: [metadata],
            page: { hasMore: true, nextCursor: "cursor-next" },
          }),
          { headers: { "content-type": "application/json" } },
        );
      }),
    );
    await expect(
      listArtifacts(api, {
        namespace: "projects",
        excludeNamespace: "skills",
        cursor: "cursor-1",
      }),
    ).resolves.toMatchObject({ items: [metadata] });
    const url = new URL(captured?.url ?? "http://invalid");
    expect(url.pathname).toBe("/v1/artifacts");
    expect(url.searchParams.get("namespace")).toBe("projects");
    expect(url.searchParams.get("excludeNamespace")).toBe("skills");
    expect(url.searchParams.get("cursor")).toBe("cursor-1");
    expect(url.searchParams.get("limit")).toBe("50");
  });

  it.each([
    ["without page information", { items: [metadata] }, "broke the public"],
    [
      "that continues without a cursor",
      { items: [metadata], page: { hasMore: true } },
      "Server returned an invalid Artifact page",
    ],
  ])("rejects a binding page %s", async (_, body, message) => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response(JSON.stringify(body), {
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(listArtifacts(api)).rejects.toMatchObject({
      code: "invalid_api_response",
      detail: expect.stringContaining(message),
    });
  });

  it("creates and updates with mutually exclusive exact CAS headers", async () => {
    const requests: Request[] = [];
    let revision = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        revision += 1;
        return response(
          JSON.stringify({
            artifact: {
              namespace: "projects",
              name: "source",
              revision: `revision-${revision}`,
            },
            mediaType: "application/zip",
            size: 3,
          }),
          {
            status: revision === 1 ? 201 : 200,
            headers: {
              "content-type": "application/json",
              ETag: `"revision-${revision}"`,
            },
          },
        );
      }),
    );
    api.csrf.replace("a".repeat(43));
    const payload = new Blob(["zip"]);
    await writeArtifact(api, {
      namespace: "projects",
      name: "source",
      mediaType: "application/zip",
      payload,
    });
    await writeArtifact(api, {
      namespace: "projects",
      name: "source",
      mediaType: "application/zip",
      payload,
      expectedRevision: "revision-1",
    });

    expect(requests[0]?.headers.get("If-None-Match")).toBe("*");
    expect(requests[0]?.headers.has("If-Match")).toBe(false);
    expect(requests[1]?.headers.get("If-Match")).toBe('"revision-1"');
    expect(requests[1]?.headers.has("If-None-Match")).toBe(false);
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[0]?.text()).resolves.toBe("zip");
  });

  it("surfaces conflict once and never retries an unsafe write", async () => {
    const fetchImplementation = vi.fn(async () =>
      response(
        JSON.stringify({
          code: "conflict",
          message: "resource state changed; retry with the current revision",
          retryable: true,
          requestId: "request-conflict",
        }),
        { status: 409, headers: { "content-type": "application/json" } },
      ),
    );
    const api = new PublicAPI(runtimeConfig, fetchImplementation);
    api.csrf.replace("a".repeat(43));
    await expect(
      writeArtifact(api, {
        namespace: "projects",
        name: "source",
        mediaType: "text/plain",
        payload: new Blob(["next"]),
        expectedRevision: "revision-stale",
      }),
    ).rejects.toMatchObject({
      code: "conflict",
      requestId: "request-conflict",
    });
    expect(fetchImplementation).toHaveBeenCalledTimes(1);
  });

  it("previews only exact bounded allowlisted UTF-8 text", async () => {
    const content = "specification {\n}";
    const exactMetadata = { ...metadata, size: content.length };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response(content, {
          headers: {
            "content-type": "text/vnd.likec4",
            "content-length": String(content.length),
          },
        }),
      ),
    );
    await expect(previewArtifact(api, exactMetadata)).resolves.toBe(content);
    expect(canPreviewArtifact(exactMetadata)).toBe(true);
    expect(
      canPreviewArtifact({ ...exactMetadata, mediaType: "text/x-diff" }),
    ).toBe(true);
    expect(
      canPreviewArtifact({
        ...exactMetadata,
        size: MAXIMUM_PREVIEW_BYTES + 1,
      }),
    ).toBe(false);
    expect(
      canPreviewArtifact({ ...exactMetadata, mediaType: "text/html" }),
    ).toBe(false);
  });

  it("rejects download metadata drift and creates a sanitized filename", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response("wrong", {
          headers: {
            "content-type": "text/plain",
            "content-length": "5",
          },
        }),
      ),
    );
    await expect(downloadArtifact(api, metadata)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    await expect(downloadedFilename(metadata.mediaType)).resolves.toBe(
      "projects-architecture-revision-2.c4",
    );
    await expect(downloadedFilename("text/x-diff")).resolves.toBe(
      "projects-architecture-revision-2.diff",
    );
  });

  it("previews an exact wordlist revision with a .txt download filename", async () => {
    const mediaType = "text/vnd.contractor.wordlist";
    const content = "first\nsecond\n";
    const wordlistMetadata = { ...metadata, mediaType, size: content.length };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        expect(new URL(request.url).searchParams.get("revision")).toBe(
          "revision-2",
        );
        expect(request.headers.get("Accept")).toBe(mediaType);
        return response(content, { headers: { "Content-Type": mediaType } });
      }),
    );
    expect(canPreviewArtifact(wordlistMetadata)).toBe(true);
    expect(
      canPreviewArtifact({
        ...wordlistMetadata,
        size: MAXIMUM_PREVIEW_BYTES + 1,
      }),
    ).toBe(false);
    await expect(downloadedFilename(mediaType)).resolves.toBe(
      "projects-architecture-revision-2.txt",
    );
    await expect(previewArtifact(api, wordlistMetadata)).resolves.toBe(content);
  });
});

describe("portable Artifact names", () => {
  it.each(nameCases.valid)("accepts %j unchanged", (name) => {
    expect(ARTIFACT_NAME_PATTERN.test(name)).toBe(true);
  });
  it.each(nameCases.invalid)("rejects %j", (name) => {
    expect(ARTIFACT_NAME_PATTERN.test(name)).toBe(false);
  });
});

const UI_ROOT = join(__dirname, "..", "..");
/**
 * The distinctive reserved Namespaces: the purpose-reserved Run Namespaces
 * (ART-R114) and the UserScope Audit standard catalog (ART-R25), the names
 * the Go (internal/artifactpolicy/namespace_literal_test.go) and Runtime
 * (runtime/tests/test_security_findings_toolset.py) scans check. inputs,
 * outputs and skills are ordinary words of other rules and are not scanned.
 * The UI names only finding-proposals; a module that needs another reserved
 * Namespace defines it next to FINDING_PROPOSAL_NAMESPACE.
 */
const RESERVED_NAMESPACES = new Set([
  FINDING_PROPOSAL_NAMESPACE,
  "contractor-system",
  "audit-standards",
]);
const NAMESPACE_DEFINITION = "src/api/artifacts.ts";
// A query key such as ["audit-standards", scheme, version] names a cache
// entry, another rule.
const QUERY_KEYS = "src/api/query-keys.ts";

/** The UI production modules: no tests, fixtures or generated code. */
function* productionModules(directory: string): Generator<string> {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      const module = relative(UI_ROOT, path).split(sep).join("/");
      if (module !== "src/api/generated" && module !== "src/test") {
        yield* productionModules(path);
      }
    } else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name)) {
      yield path;
    }
  }
}

/**
 * The lines of module `name` (relative to ui/) whose string literal spells a
 * reserved Namespace as a whole, as a comparison, ref or filter value would.
 * A path such as /v1/audit-standards/... names a route, another rule.
 */
function reservedNamespaceCopies(name: string, text: string): number[] {
  const file = ts.createSourceFile(name, text, ts.ScriptTarget.Latest, true);
  const lines: number[] = [];
  const visit = (node: ts.Node) => {
    if (
      (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) &&
      RESERVED_NAMESPACES.has(node.text) &&
      !(name === QUERY_KEYS && ts.isArrayLiteralExpression(node.parent)) &&
      !(
        name === NAMESPACE_DEFINITION &&
        ts.isVariableDeclaration(node.parent) &&
        node.parent.name.getText(file) === "FINDING_PROPOSAL_NAMESPACE"
      )
    ) {
      lines.push(file.getLineAndCharacterOfPosition(node.getStart(file)).line);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return lines.map((line) => line + 1);
}

describe("reserved Artifact Namespaces have one UI definition", () => {
  it("names the Namespace the shared reservation fixture pins", () => {
    expect(reservationCases.purposeReservedNamespaces).toContain(
      FINDING_PROPOSAL_NAMESPACE,
    );
    expect(reservationCases.purposeReservedNamespaces).toContain(
      "contractor-system",
    );
  });

  it("reports copies, not routes, query keys or the definition", () => {
    const probe =
      'if (namespace === "finding-proposals") hidden = true;\n' +
      "const system = `contractor-system`;\n" +
      'const chip = <Chip value="audit-standards" />;\n' +
      'const route = "/runs/run-1/artifacts/finding-proposals/x";\n' +
      'const path = "/v1/audit-standards/{scheme}/versions/{version}";\n' +
      'const notes = "audit-standards-notes";\n' +
      'const key = ["audit-standards", scheme] as const;\n';
    expect(reservedNamespaceCopies("src/probe.tsx", probe)).toEqual([
      1, 2, 3, 7,
    ]);
    expect(
      reservedNamespaceCopies(
        QUERY_KEYS,
        'const key = ["audit-standards", scheme] as const;\n' +
          'const ref = { namespace: "audit-standards" };\n',
      ),
    ).toEqual([2]);
    expect(
      reservedNamespaceCopies(
        NAMESPACE_DEFINITION,
        'export const FINDING_PROPOSAL_NAMESPACE = "finding-proposals";\n' +
          'export const SYSTEM = "contractor-system";\n',
      ),
    ).toEqual([2]);
  });

  it("finds no copy in UI production code", () => {
    const copies: string[] = [];
    let scanned = 0;
    for (const path of productionModules(join(UI_ROOT, "src"))) {
      const name = relative(UI_ROOT, path).split(sep).join("/");
      scanned++;
      for (const line of reservedNamespaceCopies(
        name,
        readFileSync(path, "utf8"),
      )) {
        copies.push(`${name}:${line}`);
      }
    }
    expect(scanned).toBeGreaterThan(300);
    expect(copies).toEqual([]);
  });
});
