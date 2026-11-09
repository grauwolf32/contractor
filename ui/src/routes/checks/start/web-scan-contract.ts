import type { AuditProfile } from "../../../api/audits";

/** The bundled direct-target form requires this exact published contract. */
export function isWebScanProfile(profile: AuditProfile): boolean {
  return (
    profile.ref.name === "web-scan" &&
    profile.ref.version === "1" &&
    Object.keys(profile.inputs).length === 3 &&
    profile.inputs.target?.required === true &&
    profile.inputs.engagement?.required === true &&
    profile.inputs.rulepack?.required === false &&
    ["target"].every(
      (name) =>
        profile.inputs[name]?.media_types.length === 1 &&
        profile.inputs[name]?.media_types[0] === "application/json",
    ) &&
    profile.inputs.engagement.media_types.length === 1 &&
    profile.inputs.engagement.media_types[0] ===
      "application/vnd.contractor.engagement+json" &&
    profile.inputs.rulepack.media_types.length === 2 &&
    profile.inputs.rulepack.media_types.includes("application/yaml") &&
    profile.inputs.rulepack.media_types.includes(
      "application/vnd.contractor.nuclei-rulepack+zip",
    )
  );
}
