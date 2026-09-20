import type { Audit } from "../../../api/audits";

const profileLabels: Record<string, string> = {
  "owasp-top10-2025-source-risk": "OWASP Top 10 · Source risks",
  "owasp-asvs-5-0-l1-source-review": "ASVS 5.0 · Level 1 source review",
  "owasp-wstg-4-2-source-review": "WSTG 4.2 · Source review",
  "owasp-wstg-4-2-active-http": "WSTG 4.2 · Active HTTP checks",
  "owasp-wstg-4-2-fast-source-review": "WSTG 4.2 · Fast source review",
  "owasp-wstg-4-2-fast-active-http": "WSTG 4.2 · Fast HTTP checks",
  "openapi-operation-trace": "OpenAPI · Operation trace",
  "source-checklist": "Source checklist",
};

export function auditPresetLabel(name: string): string {
  return profileLabels[name] ?? name.replaceAll("-", " ");
}

export function auditProfileLabel(audit: Audit): string {
  return auditPresetLabel(audit.profile.name);
}
