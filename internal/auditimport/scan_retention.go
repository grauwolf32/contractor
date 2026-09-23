package auditimport

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

func (i *Importer) retainScanRecovery(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
	member preparedMember,
	history []planner.ScanAttempt,
	data []byte,
	code string,
) (bool, error) {
	pkg, err := auditdomain.DecodeCheckResultPackage(data)
	if err != nil {
		return false, err
	}
	evidence := make(map[string]validatedEvidence, len(pkg.Evidence.Evidence))
	for _, item := range pkg.Evidence.Evidence {
		evidence[item.ID] = validatedEvidence{value: item}
	}
	coverage, err := semanticCoverage(config.AuditModeRiskAssessment, member.task, pkg.Results.Results[0], evidence)
	if err != nil {
		return false, err
	}
	coverage.Gaps = mergeSorted(coverage.Gaps, []string{code})
	coverage.Rationale = "Scan evidence was recovered from the durable execution journal after a Run or publication failure."
	target := contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(snapshot.Audit.AuditID), Name: auditdomain.DeterministicID("scan-recovery", execution.ExecutionID)}
	retained, err := i.artifacts.PutImmutableProject(ctx, snapshot.Audit.ProjectID, target, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: data})
	if err != nil {
		return false, err
	}
	provenance, err := json.Marshal(struct {
		Schema      string                   `json:"schema"`
		ExecutionID string                   `json:"executionId"`
		RunID       *string                  `json:"runId"`
		Task        auditstore.ExactArtifact `json:"task"`
		Manifest    auditstore.ExactArtifact `json:"executionManifest"`
	}{"contractor.audit.scan-recovery.v1", execution.ExecutionID, execution.RunID, member.member.Task, execution.Manifest})
	if err != nil {
		return false, err
	}
	links := []auditstore.ArtifactLink{{LogicalKey: "result/" + member.member.ExecutionItemID, Artifact: retained, SourceProvenance: provenance}}
	for _, item := range pkg.Evidence.Evidence {
		links = append(links, auditstore.ArtifactLink{
			LogicalKey: "evidence/" + execution.ExecutionID + "/" + item.ID, Artifact: retained,
			SourceProvenance: provenance, DisplayRef: "member:" + item.ContentMemberID,
		})
	}
	disposition := scanRecoveryDisposition(execution)
	retryable := disposition != auditstore.CollectionExecutionCancelled && !scanHistoryNeedsRecovery(history)
	item := auditstore.CollectionItem{
		ExecutionItemID: member.member.ExecutionItemID, Disposition: disposition,
		Retryable: retryable, FinalDisposition: finalDisposition(disposition), Coverage: coverage,
	}
	changed, err := i.commitCollection(ctx, claim, execution, disposition, nil, links, &code, []auditstore.CollectionItem{item})
	if errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		// A concurrent writer consumed the budget the snapshot admitted.
		return i.collectTechnical(ctx, claim, execution, []preparedMember{member},
			disposition, false, "evidence-budget-exhausted", auditstore.CoverageInconclusive)
	}
	return changed, err
}
