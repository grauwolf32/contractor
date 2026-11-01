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
  const findings: Schemas["AuditFinding"][] = [];
  const reviews: Schemas["AuditReviewRequest"][] = [];
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
    const base = build.finding(auditId, sample.id, {
      createdAt: sample.createdAt,
      updatedAt: sample.createdAt,
    });
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
    const request = build.review(
      auditId,
      `review_${sample.id}`,
      "finding-triage",
      {
        findingId: sample.id,
        subjectId: sample.id,
        subjectRevision: 1,
        createdAt: sample.createdAt,
      },
    );
    const analyst =
      sample.analyst === null
        ? undefined
        : build.decision(request, {
            verdict: "true_positive",
            severity: "high",
            rationale: sample.analyst.reason,
            actorId: "demo_analyst",
            createdAt: "2026-10-31T10:15:00Z",
          });
    const finding = build.conforming("AuditFinding", {
      ...base,
      state: sample.state as Schemas["AuditFinding"]["state"],
      firstProposal: {
        ...base.firstProposal,
        document,
        evidence: files,
        createdAt: sample.createdAt,
        origin: { ...base.firstProposal.origin, runDeleted: true },
        auditHolds: [
          {
            auditId,
            projectId,
            createdAt: sample.createdAt,
            proposal: base.firstProposal.proposal,
            evidence: files,
          },
        ],
      },
      ...(sample.assessment === "not-tested"
        ? {}
        : {
            currentAssessment: {
              assessmentId: `assessment_${sample.id}`,
              semanticAssessment: sample.assessment as NonNullable<
                Schemas["AuditFinding"]["currentAssessment"]
              >["semanticAssessment"],
              result: build.auditInput("verification"),
              receiptId: `receipt_assessment_${sample.id}`,
              directVerification: doc.http_exchange !== undefined,
              acceptedAt: sample.createdAt,
            },
          }),
      ...(analyst === undefined
        ? {}
        : {
            analystDecision: analyst,
            analystVerdict: "true_positive",
            analystSeverity: "high",
          }),
    });
    findings.push(finding);
    reviews.push(
      analyst === undefined
        ? request
        : build.conforming("AuditReviewRequest", {
            ...request,
            state: "decided",
            decision: analyst,
          }),
    );
  }
  return { projects, audits, findings, reviews, evidence };
}
