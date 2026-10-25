import { useRef, useState } from "react";
import type {
  ArtifactMetadata,
  ExactArtifactRef,
} from "../../../api/artifacts";
import type { AuditProfile, CreateAuditRequest } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { conformsTo } from "../../../api/contract";
import type { components } from "../../../api/generated/public";
import {
  downloadProjectArtifact,
  getProjectArtifactMetadata,
  writeProjectArtifact,
} from "../../../api/project-artifacts";
import type { Project } from "../../../api/projects";
import { useLaunch } from "./launch";

const RULEPACK_ZIP = "application/vnd.contractor.nuclei-rulepack+zip";

async function digest(data: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

export function useWebScanDraft(
  projectId: string,
  project: Project | undefined,
  profile: AuditProfile | undefined,
) {
  const api = usePublicAPI();
  const launch = useLaunch(projectId);
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
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<Error>();
  const [prepared, setPrepared] = useState<{
    canonical: string;
    request: CreateAuditRequest;
  }>();
  const inFlight = useRef(false);
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
  const busy = preparing || launch.pending !== undefined;
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
  if (
    (launch.startFailure?.kind === "lost" ||
      launch.startFailure?.kind === "unconfirmed") &&
    prepared?.canonical !== canonical
  )
    blocker =
      "The earlier check may already be running. Retry that request before starting another check.";
  if (
    prepared?.canonical === canonical &&
    launch.draft !== undefined &&
    launch.draft.state !== "draft"
  )
    blocker = "This check has already started. Open it from Checks.";

  async function retain(
    name: string,
    payload: Blob,
  ): Promise<ExactArtifactRef> {
    const bytes = await payload.arrayBuffer();
    const hash = await digest(bytes);
    const identity = {
      projectId,
      namespace: "scan-inputs",
      name: `${name}.${hash}`,
    };
    try {
      return (
        await writeProjectArtifact(api, {
          ...identity,
          payload,
          mediaType: payload.type,
        })
      ).artifact;
    } catch (reason) {
      // Create-only CAS may have committed before its response was lost.
      // Reconcile the content-addressed binding, including size and digest.
      try {
        const existing = await getProjectArtifactMetadata(api, identity);
        if (
          existing.size === bytes.byteLength &&
          existing.mediaType === payload.type
        ) {
          const retained = await downloadProjectArtifact(
            api,
            projectId,
            existing,
          );
          if ((await digest(await retained.blob.arrayBuffer())) === hash)
            return existing.artifact;
        }
      } catch {
        /* Keep the original write failure for the user. */
      }
      throw reason;
    }
  }

  async function submit(intent: "start" | "draft") {
    if (inFlight.current || blocker !== undefined || profile === undefined)
      return;
    inFlight.current = true;
    setPreparing(true);
    setError(undefined);
    try {
      let request =
        prepared?.canonical === canonical ? prepared.request : undefined;
      if (request === undefined) {
        const targetRef = await retain(
          "target",
          new Blob([JSON.stringify(targetDocument)], {
            type: "application/json",
          }),
        );
        const settingsRef = await retain(
          "settings",
          new Blob([JSON.stringify(settings)], { type: "application/json" }),
        );
        let rulepack = custom ? pack?.artifact : undefined;
        if (custom && file !== undefined) {
          const mediaType = file.name.toLowerCase().endsWith(".zip")
            ? RULEPACK_ZIP
            : "application/yaml";
          rulepack = await retain(
            "rulepack",
            new Blob([await file.arrayBuffer()], { type: mediaType }),
          );
        }
        request = {
          profile: { name: profile.ref.name, version: profile.ref.version },
          inputs: {
            target: targetRef,
            settings: settingsRef,
            ...(rulepack === undefined ? {} : { rulepack }),
          },
          scope: { target: url },
        };
        setPrepared({ canonical, request });
      }
      launch.launch(intent, { request, deadlineSeconds: 86400 });
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason
          : new Error("Could not save the scan inputs"),
      );
    } finally {
      inFlight.current = false;
      setPreparing(false);
    }
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
    busy,
    preparing,
    error,
    blocker,
    launch,
    request: prepared?.canonical === canonical ? prepared.request : undefined,
    submit: (intent: "start" | "draft") => void submit(intent),
  };
}

export type WebScanDraft = ReturnType<typeof useWebScanDraft>;
