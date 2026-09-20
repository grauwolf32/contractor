import { useQuery } from "@tanstack/react-query";

import { listAuditPresets } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";

export const auditModeLabels: Record<AuditProfile["mode"], string> = {
  "risk-assessment": "Risk assessment",
  "requirements-verification": "Requirements verification",
  "custom-checklist": "Custom checklist",
  "operation-tracing": "Operation tracing",
  "finding-verification": "Finding verification",
};

export function presetScope(profile: AuditProfile): string {
  if (profile.inventory.standardSelection)
    return profile.inventory.standardSelection.scope;
  switch (profile.inventory.implementation) {
    case "standard-mappings@1":
      return "Checks defined by the referenced standard.";
    case "checklist@1":
      return "Checks from your uploaded checklist.";
    case "openapi-operations@1":
      return "One check per operation in your OpenAPI document.";
    case "finding-candidates@1":
      return "Checks from your supplied finding candidates.";
  }
}

export function useAuditPresets() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["catalog", "audit-presets"],
    queryFn: ({ signal }) => listAuditPresets(api, signal),
  });
}
