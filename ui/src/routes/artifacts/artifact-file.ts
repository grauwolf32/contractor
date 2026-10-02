import { MEDIA_TYPE_PATTERN } from "../../api/artifacts";

export const SKILL_ARCHIVE_MEDIA_TYPE =
  "application/vnd.contractor.agent-skill+zip";
export const MAXIMUM_SKILL_ARCHIVE_BYTES = 16 * 1024 * 1024;

export function artifactFileStem(filename: string): string {
  const basename = filename.replace(/^.*[\\/]/, "");
  const extension = basename.lastIndexOf(".");
  const stem = extension > 0 ? basename.slice(0, extension) : basename;
  const normalized = stem
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "")
    .slice(0, 128);
  return normalized === "" ? "artifact" : normalized;
}

export function inferredArtifactMediaType(
  file: File,
  fallback: string,
  preserveFallback = false,
): string {
  // Browser MIME guesses cannot replace an existing binding's type or a type
  // the user explicitly selected before choosing the file.
  if (preserveFallback || fallback === "text/vnd.contractor.wordlist") {
    return fallback;
  }
  if (MEDIA_TYPE_PATTERN.test(file.type)) {
    return file.type;
  }
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".yaml") || lower.endsWith(".yml")) {
    return "application/yaml";
  }
  if (lower.endsWith(".json")) {
    return "application/json";
  }
  if (lower.endsWith(".md")) {
    return "text/markdown";
  }
  if (lower.endsWith(".diff") || lower.endsWith(".patch")) {
    return "text/x-diff";
  }
  if (lower.endsWith(".c4") || lower.endsWith(".likec4")) {
    return "text/vnd.likec4";
  }
  if (lower.endsWith(".zip")) {
    return "application/zip";
  }
  if (lower.endsWith(".txt")) {
    return "text/plain";
  }
  return fallback;
}
