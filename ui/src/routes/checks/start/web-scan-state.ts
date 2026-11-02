import { useState } from "react";
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { AuditProfile, CreateAuditRequest } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { conformsTo } from "../../../api/contract";
import type { components } from "../../../api/generated/public";
import { retainAuthoredMaterial } from "./authored-material";
import type { Project } from "../../../api/projects";

import {
  NUCLEI_RULEPACK_ZIP,
  rulepackMetadataError,
} from "./rulepack-metadata";

export function useWebScanDraft(
  projectId: string,
  project: Project | undefined,
  profile: AuditProfile | undefined,
) {
  const api = usePublicAPI();
  const [target, setTarget] = useState<string>();
  const [custom, setCustom] = useState(false);
  const [templateIds, setTemplateIds] = useState("");
  const [tags, setTags] = useState("");
  const [severity, setSeverity] = useState("");
  const [rate, setRate] = useState("10");
  const [pack, setPack] = useState<ArtifactMetadata>();
  const [fileSelection, setFileSelection] = useState<{
    file: File;
    key: string;
  }>();
  const file = fileSelection?.file;
  const setFile = (value: File | undefined) =>
    setFileSelection(
      value === undefined
        ? undefined
        : { file: value, key: crypto.randomUUID() },
    );
  const [prepared, setPrepared] = useState<{
    canonical: string;
    inputs: CreateAuditRequest["inputs"];
  }>();
  const url = target ?? project?.httpTarget?.url ?? "";
  const settings: components["schemas"]["AuditWebScanSettings"] = {
    schema: "contractor.audit.web-scan-settings.v1",
    templateIds,
    tags,
    severity,
    rateLimit: Number(rate),
  };
  const targetDocument = { schema: "contractor.audit.web-target.v1", url };
  const canonical = JSON.stringify({
    projectId,
    profile: profile?.ref,
    targetDocument,
    settings,
    custom,
    pack: custom && file === undefined ? pack?.artifact : undefined,
    file:
      custom && file !== undefined
        ? [file.name, file.size, file.lastModified, fileSelection?.key]
        : undefined,
  });
  let blocker: string | undefined;
  if (profile === undefined) blocker = "Loading the check type…";
  else if (!profile.serverCompatible)
    blocker = "This check type can't run on this server.";
  else if (project?.lifecycle !== "active")
    blocker = "Choose an active project.";
  else if (project.activeChecks === "prohibited")
    blocker = "Active checks are prohibited in this project's settings.";
  else if (!conformsTo("AuditWebTarget", targetDocument))
    blocker = "Enter an HTTP or HTTPS target URL.";
  else if (
    !conformsTo("AuditWebScanSettings", settings) ||
    !/^[1-9][0-9]*$/.test(rate)
  )
    blocker = "Review the rule filters and request rate.";
  else if (custom && file === undefined && pack === undefined)
    blocker = "Choose or upload your Nuclei YAML or ZIP rulepack.";
  else if (custom && file !== undefined && !/\.(ya?ml|zip)$/i.test(file.name))
    blocker = "Choose a YAML or ZIP file.";
  else if (custom && file?.size === 0) blocker = "The rule file is empty.";
  else if (custom && file !== undefined)
    blocker = rulepackMetadataError(
      file.name.toLowerCase().endsWith(".zip")
        ? NUCLEI_RULEPACK_ZIP
        : "application/yaml",
      file.size,
    );
  else if (custom && pack !== undefined)
    blocker = rulepackMetadataError(pack.mediaType, pack.size);

  function retain(name: string, payload: Blob, ensureCurrent: () => void) {
    return retainAuthoredMaterial(
      api,
      projectId,
      "scan-inputs",
      name,
      payload,
      ensureCurrent,
    );
  }

  async function prepare(ensureCurrent: () => void) {
    ensureCurrent();
    // Guard the preparation boundary as well as the visible controls: no
    // target/settings writes or file read may precede input admission.
    if (blocker !== undefined) throw new Error(blocker);
    let inputs =
      prepared?.canonical === canonical ? prepared.inputs : undefined;
    if (inputs === undefined) {
      const targetRef = await retain(
        "target",
        new Blob([JSON.stringify(targetDocument)], {
          type: "application/json",
        }),
        ensureCurrent,
      );
      const settingsRef = await retain(
        "settings",
        new Blob([JSON.stringify(settings)], { type: "application/json" }),
        ensureCurrent,
      );
      let rulepack = custom ? pack?.artifact : undefined;
      if (custom && file !== undefined) {
        const mediaType = file.name.toLowerCase().endsWith(".zip")
          ? NUCLEI_RULEPACK_ZIP
          : "application/yaml";
        rulepack = await retain(
          "rulepack",
          new Blob([await file.arrayBuffer()], { type: mediaType }),
          ensureCurrent,
        );
      }
      inputs = {
        target: targetRef,
        settings: settingsRef,
        ...(rulepack === undefined ? {} : { rulepack }),
      };
      ensureCurrent();
      setPrepared({ canonical, inputs });
    }
    return inputs;
  }
  return {
    projectId,
    project,
    target: url,
    setTarget,
    custom,
    setCustom,
    templateIds,
    setTemplateIds,
    tags,
    setTags,
    severity,
    setSeverity,
    rate,
    setRate,
    pack,
    setPack,
    file,
    setFile,
    blocker,
    inputs: prepared?.canonical === canonical ? prepared.inputs : undefined,
    prepare,
  };
}

export type WebScanDraft = ReturnType<typeof useWebScanDraft>;
