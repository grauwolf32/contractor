import type { Audit } from "../../../api/audits";

const profileLabels: Record<string, string> = {
  "owasp-top10-2025-source-risk": "OWASP Top 10 · Source risks",
  "owasp-asvs-5-0-l1-source-review": "ASVS 5.0 · Level 1 source review",
  "openapi-operation-trace": "OpenAPI · Operation trace",
  "source-checklist": "Source checklist",
};

export function auditProfileLabel(audit: Audit): string {
  return (
    profileLabels[audit.profile.name] ?? audit.profile.name.replaceAll("-", " ")
  );
}
