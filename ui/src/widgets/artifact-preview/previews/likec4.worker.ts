/// <reference lib="webworker" />

import { fromSource } from "@likec4/language-services/browser";

import type { LikeC4WorkerResponse } from "./likec4-protocol";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "LikeC4 rendering failed";
}

function stripIcons(value: unknown, visited = new WeakSet<object>()): void {
  if (value === null || typeof value !== "object" || visited.has(value)) {
    return;
  }
  visited.add(value);
  if (!Array.isArray(value)) {
    delete (value as { icon?: unknown }).icon;
  }
  for (const child of Object.values(value)) {
    stripIcons(child, visited);
  }
}

self.onmessage = async (event: MessageEvent<string>) => {
  let likec4: Awaited<ReturnType<typeof fromSource>> | undefined;
  try {
    likec4 = await fromSource(event.data);
    const errors = likec4.getErrors();
    if (errors.length > 0) {
      const details = errors
        .slice(0, 8)
        .map((error) => `Line ${error.line}: ${error.message}`)
        .join("\n");
      const suffix =
        errors.length > 8 ? `\n…and ${errors.length - 8} more` : "";
      self.postMessage({
        ok: false,
        message: `${details}${suffix}`,
      } satisfies LikeC4WorkerResponse);
      return;
    }

    const engine = likec4;
    const models = await Promise.all(
      engine.projects().map((project) => engine.layoutedModel(project)),
    );
    for (const model of models) {
      // LikeC4 otherwise renders URL and data URI icons as images. Artifacts
      // are untrusted, so inline previews intentionally omit every icon.
      stripIcons(model.$data);
    }
    self.postMessage({
      ok: true,
      models: models.map((model) => model.$data),
    } satisfies LikeC4WorkerResponse);
  } catch (error) {
    self.postMessage({
      ok: false,
      message: errorMessage(error),
    } satisfies LikeC4WorkerResponse);
  } finally {
    await likec4?.dispose();
  }
};
