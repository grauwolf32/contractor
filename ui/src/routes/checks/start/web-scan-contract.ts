import type { AuditProfile } from "../../../api/audits";

/** The bundled direct-target form requires this exact published contract. */
export function isWebScanProfile(profile: AuditProfile): boolean {
  return (
    profile.ref.name === "web-scan" &&
    profile.ref.version === "1" &&
    profile.inventory.implementation === "web-scans@1" &&
    Object.keys(profile.inputs).length === 3 &&
    profile.inputs.target?.required === true &&
    profile.inputs.settings?.required === true &&
    profile.inputs.rulepack?.required === false &&
    ["target", "settings"].every(
      (name) =>
        profile.inputs[name]?.mediaTypes.length === 1 &&
        profile.inputs[name]?.mediaTypes[0] === "application/json",
    ) &&
    profile.inputs.rulepack.mediaTypes.length === 2 &&
    profile.inputs.rulepack.mediaTypes.includes("application/yaml") &&
    profile.inputs.rulepack.mediaTypes.includes(
      "application/vnd.contractor.nuclei-rulepack+zip",
    )
  );
}
