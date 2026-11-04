// Synthetic review data from the preserved, user-reviewed OpenDesign trial.
// Test/browser fixtures only; production routes never import this module.
// Synthetic review data from the preserved, user-reviewed OpenDesign trial.
// Test/browser fixtures only; production routes never import this module.
import examples from "../../../docs/design/ui/experiments/issues-review-opendesign-2026-11-01/examples.json" with { type: "json" };
import type { components } from "../api/generated/public";
import * as build from "./builders";

type Schemas = components["schemas"];
export function issueReviewFixtures() {
  const projects = [
    build.project("project_shop", { name: "Shop API" }),
    build.project("project_records", { name: "Records API" }),
  ];
  const audits: Schemas["Audit"][] = [];
  const findings: Schemas["ProjectFinding"][] = [];
  const sources: Schemas["FindingProposalReceipt"][] = [];
  const evidence: {
    projectId: string;
    metadata: Schemas["ArtifactMetadata"];
    text: string;
  }[] = [];
  for (const sample of examples.examples) {
    const projectId =
      sample.project === "Records API" ? "project_records" : "project_shop";
    const auditId = `audit_${projectId}_${sample.check.toLowerCase().replaceAll(/[^a-z0-9]+/g, "_")}`;
    if (!audits.some((audit) => audit.auditId === auditId))
      audits.push(
        build.audit(projectId, auditId, "completed", {
          profile: {
            name: sample.check.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-"),
            version: "1",
            digest: build.DIGEST,
          },
          inputs: {
            source: build.auditInput("shop-source", {
              ref: {
                namespace: "sources",
                name: "shop-source",
                revision: "source-r1",
              },
            }),
          },
          holdState: "held",
          dispatchState: "closed",
        }),
      );
    const doc = build.conforming(
      "FindingProposalDocument",
      sample.proposal as Schemas["FindingProposalDocument"],
    );
    const document = {
      ...doc,
      description: `${doc.description}\n\n## Impact\n${sample.impact}`,
    };
    const files = sample.files.map((file) => {
      const match = /^([^/]+)\/(.+)@([^@]+)$/.exec(file.ref);
      if (match === null)
        throw new Error("Invalid exact synthetic evidence identity");
      const ref = {
        namespace: match[1]!,
        name: match[2]!,
        revision: match[3]!,
      };
      const sizeBytes = new TextEncoder().encode(file.text).length;
      evidence.push({
        projectId,
        metadata: build.artifactMetadata(ref.namespace, ref.name, {
          artifact: ref,
          mediaType: file.mediaType,
          size: sizeBytes,
          frozen: true,
          canUploadNewVersion: false,
        }),
        text: file.text,
      });
      return {
        ref,
        digest: build.DIGEST,
        mediaType: file.mediaType,
        sizeBytes,
      };
    });

    const claim: Schemas["FindingClaim"] = {
      title: document.title,
      description: document.description,
      subject: document.subject,
      preconditions: document.preconditions,
      standard_refs: document.standard_refs,
      severity_suggestion: document.severity_suggestion,
      limitations: document.limitations,
      ...(document.hypothesis === undefined
        ? {}
        : { hypothesis: document.hypothesis }),
      ...(document.locations === undefined
        ? {}
        : { locations: document.locations }),
      ...(document.http_exchange === undefined
        ? {}
        : { http_exchange: document.http_exchange }),
    };
    const finding = build.finding(
      projectId,
      sample.id,
      {
        createdAt: sample.createdAt,
        updatedAt: sample.createdAt,
        ...(sample.analyst === null
          ? {}
          : {
              state: "confirmed",
              analystVerdict: "true_positive",
              analystSeverity: "high",
              rationale: sample.analyst.reason,
              actorId: "demo_analyst",
              assessedAt: "2026-10-31T10:15:00Z",
            }),
      },
      claim,
    );
    findings.push(finding);
    sources.push(
      build.findingSource(
        finding,
        { evidence: files, projectEvidence: files },
        document,
      ),
    );
  }
  return { projects, audits, findings, sources, evidence };
}
