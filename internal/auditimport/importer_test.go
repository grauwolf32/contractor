package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestImporterAcceptsExactFrozenResultAndEvidence(t *testing.T) {
	harness := newImportHarness(t)
	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	if err != nil || !worked {
		t.Fatalf("collect accepted result = (%t, %v)", worked, err)
	}
	collected := harness.store.collected
	if collected.Disposition != auditstore.CollectionAccepted || collected.SourceOutput == nil ||
		len(collected.Items) != 1 || collected.Items[0].Result == nil ||
		collected.Items[0].Coverage.Status != auditstore.CoverageSatisfied ||
		len(collected.Retained) != 2 || collected.RequestDigest == "" {
		t.Fatalf("accepted collection = %+v", collected)
	}
	if collected.Items[0].Result.Ref.Namespace != deterministicID("audit", harness.snapshot.Audit.AuditID) {
		t.Fatalf("accepted result was not retained in protected namespace: %+v", collected.Items[0].Result)
	}
}

func TestImporterRecordsInvalidAndMissingSucceededOutputs(t *testing.T) {
	t.Run("invalid", func(t *testing.T) {
		harness := newImportHarness(t)
		harness.artifacts.runPayload = []byte("not a package")
		harness.artifacts.runDescriptor.Digest = digestBytes(harness.artifacts.runPayload)
		harness.artifacts.runDescriptor.SizeBytes = int64(len(harness.artifacts.runPayload))
		worked, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
		if err != nil || !worked || harness.store.collected.Disposition != auditstore.CollectionInvalidResult ||
			harness.store.collected.ErrorCode == nil || !harness.store.collected.Items[0].Retryable {
			t.Fatalf("invalid collection = (%t, %v, %+v)", worked, err, harness.store.collected)
		}
	})
	t.Run("missing", func(t *testing.T) {
		harness := newImportHarness(t)
		harness.artifacts.missingBinding = true
		worked, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
		if err != nil || !worked || harness.store.collected.Disposition != auditstore.CollectionMissingOutput ||
			harness.store.collected.SourceOutput != nil || !harness.store.collected.Items[0].Retryable {
			t.Fatalf("missing collection = (%t, %v, %+v)", worked, err, harness.store.collected)
		}
	})
}

func TestImporterRecordsFailedAndCancelledWithoutInventedOutput(t *testing.T) {
	for _, test := range []struct {
		name        string
		outcome     auditstore.TerminalOutcome
		disposition auditstore.CollectionDisposition
		retryable   bool
	}{
		{"failed", auditstore.TerminalFailed, auditstore.CollectionExecutionFailed, true},
		{"cancelled", auditstore.TerminalCancelled, auditstore.CollectionExecutionCancelled, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			harness := newImportHarness(t)
			harness.execution.TerminalOutcome = &test.outcome
			worked, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
			if err != nil || !worked || harness.store.collected.Disposition != test.disposition ||
				harness.store.collected.SourceOutput != nil || harness.store.collected.Items[0].Retryable != test.retryable ||
				harness.store.collected.Items[0].Result != nil {
				t.Fatalf("technical collection = (%t, %v, %+v)", worked, err, harness.store.collected)
			}
		})
	}
}

func TestImporterRetainsAuditChildFindingProposalsBeforeCollection(t *testing.T) {
	harness := newImportHarness(t)
	profile := loadResultProfileWithFindingConfirmation(t, "human-required")
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Audit.Profile = auditstore.ProfileIdentity{
		Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
	}
	harness.snapshot.Audit.ProfileSnapshot = profileSnapshot
	revision := "finding-revision"
	findings := &fakeFindingRetention{receipts: []findingintake.Receipt{{
		ReceiptID: "finding-receipt", Proposal: findingintake.ExactArtifact{Ref: contracts.ArtifactRef{
			Namespace: "finding-proposals", Name: "candidate", Revision: &revision,
		}},
		Origin: findingintake.Origin{
			RunID: *harness.execution.RunID,
			Audit: &findingintake.AuditOrigin{
				AuditID: harness.execution.AuditID, ExecutionID: harness.execution.ExecutionID,
				Role: string(harness.execution.Role),
			},
		},
	}}}
	harness.importer, err = New(harness.store, harness.importer.runs, harness.artifacts, findings)
	if err != nil {
		t.Fatal(err)
	}
	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	if err != nil || !worked || len(findings.imports) != 1 ||
		findings.imports[0].AuditID != harness.execution.AuditID ||
		harness.store.collected.Disposition != auditstore.CollectionAccepted {
		t.Fatalf("finding-aware collection = (%t, %v, imports=%+v, collection=%+v)",
			worked, err, findings.imports, harness.store.collected)
	}
}

func TestImporterAssociatesOnlyExactInvocationLocalFindingProposal(t *testing.T) {
	harness := newImportHarness(t)
	profile := loadResultProfileWithFindingConfirmation(t, "human-required")
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Audit.Profile = auditstore.ProfileIdentity{
		Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
	}
	harness.snapshot.Audit.ProfileSnapshot = profileSnapshot

	proposalRevision := "proposal-r1"
	proposal := findingintake.ResolvedProposal{
		ReceiptID: "finding-receipt", Proposal: findingintake.ExactArtifact{
			Ref: contracts.ArtifactRef{
				Namespace: "audit-findings", Name: "candidate", Revision: &proposalRevision,
			},
			Digest: digestBytes([]byte("proposal")), MediaType: "application/json", SizeBytes: 8,
		},
	}
	findings := &fakeFindingRetention{resolved: []findingintake.ResolvedProposal{proposal}}
	harness.importer, err = New(harness.store, harness.importer.runs, harness.artifacts, findings)
	if err != nil {
		t.Fatal(err)
	}
	rebuildHarnessResult(t, &harness, []auditdomain.ProposalSelection{{
		InvocationID: "worker-invocation-1", ClientKey: "candidate-1",
	}})

	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	if err != nil || !worked {
		t.Fatalf("collect proposal association = (%t, %v)", worked, err)
	}
	associations := harness.store.collected.Items[0].FindingAssociations
	if len(associations) != 1 || associations[0].ReceiptID != proposal.ReceiptID ||
		associations[0].Proposal.Digest != proposal.Proposal.Digest ||
		associations[0].SemanticAssessment != "satisfied" {
		t.Fatalf("finding associations = %+v", associations)
	}
}

func TestImporterRejectsUnexpectedProposalWhenFindingsAreDisabled(t *testing.T) {
	harness := newImportHarness(t)
	revision := "finding-revision"
	findings := &fakeFindingRetention{receipts: []findingintake.Receipt{{
		ReceiptID: "unexpected-receipt", Proposal: findingintake.ExactArtifact{Ref: contracts.ArtifactRef{
			Namespace: "finding-proposals", Name: "unexpected", Revision: &revision,
		}},
	}}}
	runs := harness.importer.runs
	var err error
	harness.importer, err = New(harness.store, runs, harness.artifacts, findings)
	if err != nil {
		t.Fatal(err)
	}
	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	if err != nil || !worked || len(findings.imports) != 0 ||
		harness.store.collected.Disposition != auditstore.CollectionContractInvalid {
		t.Fatalf("disabled finding collection = (%t, %v, imports=%+v, collection=%+v)",
			worked, err, findings.imports, harness.store.collected)
	}
}

func TestImporterConvertsPermanentPinnedContractFailureToReceipt(t *testing.T) {
	harness := newImportHarness(t)
	corrupt := []byte("not an Audit task package")
	harness.artifacts.project[refKey(harness.store.members[0].Task.Ref)] = corrupt
	digest := digestBytes(corrupt)
	harness.store.members[0].Task.Digest = digest
	harness.snapshot.Items[0].Task.Digest = digest

	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	collected := harness.store.collected
	if err != nil || !worked || collected.Disposition != auditstore.CollectionContractInvalid ||
		collected.SourceOutput != nil || collected.ErrorCode == nil ||
		*collected.ErrorCode != "collection-contract-invalid" || len(collected.Items) != 1 ||
		collected.Items[0].Retryable || collected.Items[0].FinalDisposition != auditstore.FinalInvalidResult ||
		collected.Items[0].Coverage.Status != auditstore.CoverageBlocked {
		t.Fatalf("contract-invalid collection = (%t, %v, %+v)", worked, err, collected)
	}
}

func TestImporterSettlesEvidenceBudgetExhaustionWithoutStaging(t *testing.T) {
	harness := newImportHarness(t)
	harness.snapshot.Audit.Limits.MaxEvidenceBytes = harness.artifacts.runDescriptor.SizeBytes - 1
	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	if err != nil || !worked || harness.store.collected.Disposition != auditstore.CollectionInvalidResult ||
		harness.store.collected.ErrorCode == nil || *harness.store.collected.ErrorCode != "evidence-budget-exhausted" ||
		harness.store.collected.Items[0].Retryable || len(harness.store.collected.Retained) != 0 {
		t.Fatalf("evidence budget collection = (%t, %v, %+v)", worked, err, harness.store.collected)
	}
}

func TestImporterFinalizesTruthfulReportWithZeroDenominator(t *testing.T) {
	profile := loadResultProfile(t)
	profileSnapshot, _ := json.Marshal(profile)
	roundID := "round-report"
	taskRevision := "task-report"
	task := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "task", Revision: &taskRevision},
		Digest: digestBytes([]byte("task")), MediaType: auditdomain.PackageMediaType, SizeBytes: 4,
	}
	disposition := auditstore.FinalExcluded
	item := auditstore.Item{
		ItemID: "item-report", AuditID: "audit-report", RoundID: roundID,
		ItemKey: "check-report", Ordinal: 0, Kind: "checklist", SubjectKey: "check-report",
		Task: task, WorkflowRole: "check", State: auditstore.ItemSettled,
		FinalDisposition: &disposition,
	}
	coverage := auditstore.CoverageRow{
		AuditID: item.AuditID, RoundID: roundID, ItemID: item.ItemID,
		Ordinal: 0, ItemKey: item.ItemKey, SubjectKey: item.SubjectKey,
		Coverage: auditstore.Coverage{
			Status: auditstore.CoverageExcluded, Requested: []string{"source"},
			Completed: []string{}, Gaps: []string{"policy-excluded"},
		},
	}
	proposalDocument := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-report",
		Title: "Untrusted redirect target", Description: "A redirect target may cross the intended origin.",
		Subject:       auditdomain.FindingSubject{Kind: "component", Key: "redirect-handler"},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
		SeveritySuggestion: "medium", Limitations: []string{"Dynamic behavior was not exercised."},
	}
	proposalBytes, err := auditdomain.EncodeFindingProposal(proposalDocument)
	if err != nil {
		t.Fatal(err)
	}
	proposalRevision := "proposal-report-r1"
	proposalArtifact := auditstore.ExactArtifact{
		Ref: contracts.ArtifactRef{
			Namespace: "audit-test", Name: "proposal-report", Revision: &proposalRevision,
		},
		Digest: digestBytes(proposalBytes), MediaType: "application/json", SizeBytes: int64(len(proposalBytes)),
	}
	analystSeverity := "high"
	store := &fakeImportStore{
		items: []auditstore.Item{item}, coverage: []auditstore.CoverageRow{coverage},
		counts: auditstore.CollectionDispositionCounts{ExecutionFailed: 1},
		findings: []auditstore.ReportFinding{
			{FindingID: "finding-confirmed", State: "confirmed", FirstProposal: proposalArtifact,
				Revision: 3, Decision: &auditstore.ReportFindingDecision{
					DecisionID: "decision-report", ActorID: "owner-report", Verdict: "true_positive",
					Severity: &analystSeverity, Rationale: "Confirmed against retained evidence.",
					SubjectRevision: 2, SubjectDigest: digestBytes([]byte("finding-subject")), CreatedAt: time.Unix(90, 0),
				}},
			{FindingID: "finding-proposed", State: "proposed", FirstProposal: proposalArtifact, Revision: 1},
		},
	}
	artifactAccess := &fakeImportArtifacts{
		project: map[string][]byte{refKey(proposalArtifact.Ref): proposalBytes}, writes: map[string][]byte{},
	}
	importer, err := New(store, &fakeImportRuns{}, artifactAccess)
	if err != nil {
		t.Fatal(err)
	}
	manifestRevision := "round-manifest"
	snapshot := auditstore.ReconcileSnapshot{
		Audit: auditstore.Audit{
			AuditID: item.AuditID, ProjectID: "project-report",
			Profile:          auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
			ProfileSnapshot:  profileSnapshot,
			BaselineSnapshot: json.RawMessage(`{"schema":"contractor.audit.baseline.v1","inventory":{"gaps":[]}}`),
			State:            auditstore.AuditFinalizing, Revision: 7, UpdatedAt: time.Unix(100, 0),
		},
		Round: &auditstore.Round{
			RoundID: roundID, AuditID: item.AuditID, Ordinal: 1,
			Manifest: auditstore.ExactArtifact{
				Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "round", Revision: &manifestRevision},
				Digest: digestBytes([]byte("round")),
			},
			State: auditstore.RoundClosed, ExpectedItemCount: 1, Revision: 3,
		},
	}
	claim := auditstore.ControllerClaim{AuditID: item.AuditID, HolderID: "holder", Epoch: 2}
	worked, err := importer.Finalize(context.Background(), claim, snapshot)
	if err != nil || !worked || store.committed.Machine.Artifact.Ref.Revision == nil ||
		store.committed.Summary.Artifact.Ref.Revision == nil {
		t.Fatalf("finalize report = (%t, %v, %+v)", worked, err, store.committed)
	}
	machineBytes := artifactAccess.writes["report.json"]
	if !containsBytes(machineBytes, `"conclusion":"completed-with-gaps"`) ||
		!containsBytes(machineBytes, `"zeroDenominator":true`) ||
		!containsBytes(machineBytes, `"executionFailed":1`) ||
		!containsBytes(machineBytes, `"confirmed":[{"findingId":"finding-confirmed"`) ||
		!containsBytes(machineBytes, `"proposed":[{"findingId":"finding-proposed"`) ||
		!containsBytes(machineBytes, `"severitySuggestion":"medium"`) ||
		!containsBytes(machineBytes, `"verdict":"true_positive"`) {
		t.Fatalf("machine report is not truthful: %s", machineBytes)
	}
	if !containsBytes(artifactAccess.writes["report.txt"], "Findings: confirmed=1 proposed=1") ||
		!containsBytes(artifactAccess.writes["report.txt"], "not a security or compliance certification") {
		t.Fatalf("human summary omitted qualification: %s", artifactAccess.writes["report.txt"])
	}
}

func TestOperationCoverageKeepsTraceStateSeparateFromAssessment(t *testing.T) {
	task := auditdomain.ItemTask{Operation: &auditdomain.OperationTask{Gaps: []string{}}}
	result := auditdomain.CheckResult{
		Assessment: "inconclusive",
		Coverage: auditdomain.ResultCoverage{
			Requested: []string{"operation-resolution"},
			Completed: []string{"operation-resolution"}, Gaps: []string{},
		},
	}
	coverage, err := semanticCoverage(config.AuditModeOperationTracing, task, result, map[string]validatedEvidence{})
	if err != nil || coverage.Status != auditstore.CoverageTracedComplete {
		t.Fatalf("complete operation coverage = (%+v, %v)", coverage, err)
	}

	task.Operation.Gaps = []string{"callback-not-selected"}
	coverage, err = semanticCoverage(config.AuditModeOperationTracing, task, result, map[string]validatedEvidence{})
	if err != nil || coverage.Status != auditstore.CoverageTracedPartial {
		t.Fatalf("partial operation coverage = (%+v, %v)", coverage, err)
	}
	result.Coverage.Completed = []string{}
	coverage, err = semanticCoverage(config.AuditModeOperationTracing, task, result, map[string]validatedEvidence{})
	if err != nil || coverage.Status != auditstore.CoverageUnmapped {
		t.Fatalf("unmapped operation coverage = (%+v, %v)", coverage, err)
	}
	result.Assessment = "not-tested"
	result.Coverage.Completed = []string{"operation-resolution"}
	if _, err := semanticCoverage(config.AuditModeOperationTracing, task, result, map[string]validatedEvidence{}); err == nil {
		t.Fatal("not-tested operation accepted completed trace coverage")
	}
}

func TestChecklistCoverageDoesNotClaimConclusiveViolationAcrossGaps(t *testing.T) {
	task := auditdomain.ItemTask{Checklist: &auditdomain.ChecklistTask{
		RequiredEvidence: []string{"source"},
	}}
	result := auditdomain.CheckResult{
		Assessment: "violated", EvidenceIDs: []string{"evidence"},
		Coverage: auditdomain.ResultCoverage{
			Requested: []string{"source"}, Completed: []string{"source"},
			Gaps: []string{"unresolved-helper"},
		},
	}
	evidence := map[string]validatedEvidence{
		"evidence": {value: auditdomain.Evidence{ID: "evidence", Kind: "source"}},
	}
	coverage, err := semanticCoverage(config.AuditModeCustomChecklist, task, result, evidence)
	if err != nil || coverage.Status != auditstore.CoverageInconclusive {
		t.Fatalf("gapped violation coverage = (%+v, %v)", coverage, err)
	}
}

func TestReportCoverageCountsOperationStatesWithoutClaimingPartialCompletion(t *testing.T) {
	rows := []auditstore.CoverageRow{
		{Coverage: auditstore.Coverage{Status: auditstore.CoverageTracedComplete}},
		{Coverage: auditstore.Coverage{Status: auditstore.CoverageTracedPartial}},
		{Coverage: auditstore.Coverage{Status: auditstore.CoverageUnmapped}},
	}
	summary := summarizeCoverage(rows)
	if summary.Counts.TracedComplete != 1 || summary.Counts.TracedPartial != 1 || summary.Counts.Unmapped != 1 ||
		summary.ApplicableDenominator != 3 || summary.AssessedApplicable != 1 ||
		summary.AssessedPercent == nil || *summary.AssessedPercent < 33 || *summary.AssessedPercent > 34 ||
		!hasIncompleteCoverage(summary.Counts) {
		t.Fatalf("operation coverage summary = %+v", summary)
	}
}

type importHarness struct {
	importer  *Importer
	store     *fakeImportStore
	artifacts *fakeImportArtifacts
	claim     auditstore.ControllerClaim
	snapshot  auditstore.ReconcileSnapshot
	execution auditstore.Execution
}

func newImportHarness(t *testing.T) importHarness {
	t.Helper()
	profile := loadResultProfile(t)
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	sourceRevision := "source-r1"
	taskRevision := "task-r1"
	manifestRevision := "manifest-r1"
	outputRevision := "output-r1"
	sourceRef := contracts.ArtifactRef{Namespace: "inputs", Name: "checklist", Revision: &sourceRevision}
	taskDocument := auditdomain.ItemTask{
		Schema: auditdomain.TaskSchema, ItemKey: "check-1", Kind: "checklist",
		SubjectKey: "check-1", WorkflowRole: "check",
		SourceContentDigest: digestBytes([]byte("checklist")), SourceMediaType: "application/json",
		SourceRef: sourceRef, CanonicalInventoryDigest: digestBytes([]byte("inventory")),
		Checklist: &auditdomain.ChecklistTask{
			Version: "1", Statement: "Verify source evidence", Applicability: "always",
			AllowedMethods: []string{"static"}, RequiredEvidence: []string{"source"}, ReviewPolicy: "automatic",
		},
	}
	taskJSON, err := auditdomain.EncodeItemTask(taskDocument)
	if err != nil {
		t.Fatal(err)
	}
	taskPayload, taskPackage, err := auditdomain.BuildPackage("task-1", auditdomain.PackageKindTask, "", []auditdomain.PackageInput{{
		ID: "task-document", Path: "task.json", MediaType: "application/json", Data: taskJSON,
	}})
	if err != nil {
		t.Fatal(err)
	}
	task := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "task-1", Revision: &taskRevision},
		Digest: taskPackage.Digest, MediaType: auditdomain.PackageMediaType, SizeBytes: int64(len(taskPayload)),
	}
	manifest := auditdomain.ExecutionManifest{Schema: auditdomain.ExecutionManifestSchema, Items: []auditdomain.ExecutionItem{{
		ItemKey: "check-1", Ordinal: 0, SubjectKey: "check-1",
		TaskPackageID: "task-1", TaskPackageDigest: task.Digest, TaskRef: &task.Ref,
		Inputs: []auditdomain.ExactInput{{Name: "source", Ref: sourceRef, Digest: taskDocument.SourceContentDigest}},
	}}}
	manifestBytes, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestDescriptor := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "manifest", Revision: &manifestRevision},
		Digest: digestBytes(manifestBytes), MediaType: "application/json", SizeBytes: int64(len(manifestBytes)),
	}
	resultSet, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: manifestDescriptor.Digest,
		Results: []auditdomain.CheckResult{{
			ItemKey: "check-1", SubjectKey: "check-1", Assessment: "satisfied",
			Summary: "The required evidence is present.", EvidenceIDs: []string{"ev-1"},
			Coverage:  auditdomain.ResultCoverage{Requested: []string{"source"}, Completed: []string{"source"}, Gaps: []string{}},
			Proposals: []auditdomain.ProposalSelection{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := auditdomain.EncodeEvidence(auditdomain.EvidenceEnvelope{
		Schema: auditdomain.EvidenceSchema, Evidence: []auditdomain.Evidence{{
			ID: "ev-1", Kind: "source", Summary: "Static source evidence", ContentMemberID: "ev-body",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resultPayload, resultPackage, err := auditdomain.BuildPackage("result-1", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
		{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: "application/json", Data: resultSet},
		{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: "application/json", Data: evidence},
		{ID: "ev-body", Path: "evidence/source.txt", MediaType: "text/plain", Data: []byte("evidence")},
	})
	if err != nil {
		t.Fatal(err)
	}
	output := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "outputs", Name: "result", Revision: &outputRevision},
		Digest: resultPackage.Digest, MediaType: auditdomain.PackageMediaType, SizeBytes: int64(len(resultPayload)),
	}
	roundID := "round-1"
	runID := "run-1"
	outcome := auditstore.TerminalSucceeded
	execution := auditstore.Execution{
		ExecutionID: "execution-1", AuditID: "audit-1", RoundID: &roundID,
		Role: auditstore.ExecutionCheck, Manifest: manifestDescriptor,
		RunID: &runID, State: auditstore.ExecutionCollecting, TerminalOutcome: &outcome,
	}
	member := auditstore.ExecutionItem{
		ExecutionItemID: "execution-item-1", ExecutionID: execution.ExecutionID,
		AuditID: execution.AuditID, RoundID: roundID, ItemID: "item-1",
		BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{},
		State: auditstore.ItemCollecting,
	}
	item := auditstore.Item{
		ItemID: member.ItemID, AuditID: execution.AuditID, RoundID: roundID,
		ItemKey: "check-1", Ordinal: 0, Kind: "checklist", SubjectKey: "check-1",
		Task: task, WorkflowRole: "check", State: auditstore.ItemCollecting,
	}
	snapshot := auditstore.ReconcileSnapshot{Audit: auditstore.Audit{
		AuditID: execution.AuditID, OwnerID: "owner-1", ProjectID: "project-1",
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: profileSnapshot, State: auditstore.AuditActive,
		Limits: auditstore.Limits{MaxEvidenceBytes: 1 << 30},
	}, Items: []auditstore.Item{item}, Executions: []auditstore.Execution{execution}}
	artifactAccess := &fakeImportArtifacts{
		project:       map[string][]byte{refKey(task.Ref): taskPayload, refKey(manifestDescriptor.Ref): manifestBytes},
		runDescriptor: output, runPayload: resultPayload, frozen: true,
	}
	store := &fakeImportStore{members: []auditstore.ExecutionItem{member}}
	runSnapshot, _ := json.Marshal(profile.Workflows["check"].Workflow)
	runs := &fakeImportRuns{run: runstore.WorkflowRun{
		RunID: runID, ProjectID: stringPointer("project-1"),
		WorkflowName:          profile.Workflows["check"].Workflow.Ref.Name,
		WorkflowVersion:       profile.Workflows["check"].Workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: runSnapshot,
		PublicationMode: runstore.PublicationAuditManaged, AuditExecutionID: stringPointer(execution.ExecutionID),
		State: runstore.RunSucceeded,
	}}
	importer, err := New(store, runs, artifactAccess)
	if err != nil {
		t.Fatal(err)
	}
	return importHarness{
		importer: importer, store: store, artifacts: artifactAccess,
		claim:    auditstore.ControllerClaim{AuditID: execution.AuditID, HolderID: "holder", Epoch: 1},
		snapshot: snapshot, execution: execution,
	}
}

type fakeImportStore struct {
	members   []auditstore.ExecutionItem
	collected auditstore.CollectParams
	items     []auditstore.Item
	coverage  []auditstore.CoverageRow
	findings  []auditstore.ReportFinding
	counts    auditstore.CollectionDispositionCounts
	committed auditstore.CommitReportParams
}

func (f *fakeImportStore) ListExecutionItems(context.Context, string) ([]auditstore.ExecutionItem, error) {
	return append([]auditstore.ExecutionItem{}, f.members...), nil
}
func (f *fakeImportStore) ListItems(context.Context, string) ([]auditstore.Item, error) {
	return append([]auditstore.Item{}, f.items...), nil
}
func (f *fakeImportStore) ListCoverage(_ context.Context, _, _ string, after, limit int) ([]auditstore.CoverageRow, error) {
	result := make([]auditstore.CoverageRow, 0, limit)
	for _, row := range f.coverage {
		if row.Ordinal > after && len(result) < limit {
			result = append(result, row)
		}
	}
	return result, nil
}
func (f *fakeImportStore) CollectionDispositionCounts(context.Context, string) (auditstore.CollectionDispositionCounts, error) {
	return f.counts, nil
}
func (f *fakeImportStore) ListReportFindings(context.Context, string) ([]auditstore.ReportFinding, error) {
	return append([]auditstore.ReportFinding{}, f.findings...), nil
}
func (f *fakeImportStore) Collect(_ context.Context, params auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error) {
	f.collected = params
	return auditstore.CollectionReceipt{}, true, nil
}
func (f *fakeImportStore) CommitReport(_ context.Context, params auditstore.CommitReportParams) (auditstore.Audit, error) {
	f.committed = params
	return auditstore.Audit{State: auditstore.AuditCompleted}, nil
}

type fakeImportRuns struct{ run runstore.WorkflowRun }

func (f *fakeImportRuns) GetRun(context.Context, string) (runstore.WorkflowRun, error) {
	return f.run, nil
}

type fakeFindingRetention struct {
	receipts []findingintake.Receipt
	imports  []findingintake.ImportRequest
	resolved []findingintake.ResolvedProposal
}

func (f *fakeFindingRetention) ResolveAuditProposals(
	context.Context, string, string, string, string, []findingintake.ProposalKey,
) ([]findingintake.ResolvedProposal, error) {
	return append([]findingintake.ResolvedProposal(nil), f.resolved...), nil
}

func (f *fakeFindingRetention) ListRun(
	context.Context, string, string, findingintake.ListQuery,
) ([]findingintake.Receipt, error) {
	return append([]findingintake.Receipt(nil), f.receipts...), nil
}

func (f *fakeFindingRetention) ImportIntoAudit(
	_ context.Context, request findingintake.ImportRequest,
) (findingintake.AuditHold, bool, error) {
	f.imports = append(f.imports, request)
	return findingintake.AuditHold{AuditID: request.AuditID}, false, nil
}

type fakeImportArtifacts struct {
	project        map[string][]byte
	runDescriptor  auditstore.ExactArtifact
	runPayload     []byte
	frozen         bool
	missingBinding bool
	writes         map[string][]byte
}

func rebuildHarnessResult(
	t *testing.T, harness *importHarness, proposals []auditdomain.ProposalSelection,
) {
	t.Helper()
	manifestBytes := harness.artifacts.project[refKey(harness.execution.Manifest.Ref)]
	resultSet, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: digestBytes(manifestBytes),
		Results: []auditdomain.CheckResult{{
			ItemKey: "check-1", SubjectKey: "check-1", Assessment: "satisfied",
			Summary: "The required evidence is present.", EvidenceIDs: []string{"ev-1"},
			Coverage: auditdomain.ResultCoverage{
				Requested: []string{"source"}, Completed: []string{"source"}, Gaps: []string{},
			},
			Proposals: proposals,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := auditdomain.EncodeEvidence(auditdomain.EvidenceEnvelope{
		Schema: auditdomain.EvidenceSchema, Evidence: []auditdomain.Evidence{{
			ID: "ev-1", Kind: "source", Summary: "Static source evidence", ContentMemberID: "ev-body",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	payload, pkg, err := auditdomain.BuildPackage(
		"result-1", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
			{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: "application/json", Data: resultSet},
			{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: "application/json", Data: evidence},
			{ID: "ev-body", Path: "evidence/source.txt", MediaType: "text/plain", Data: []byte("evidence")},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	harness.artifacts.runPayload = payload
	harness.artifacts.runDescriptor.Digest = pkg.Digest
	harness.artifacts.runDescriptor.SizeBytes = int64(len(payload))
}

func (f *fakeImportArtifacts) ReadProjectExact(_ context.Context, _ string, artifact auditstore.ExactArtifact) ([]byte, error) {
	value, exists := f.project[refKey(artifact.Ref)]
	if !exists || digestBytes(value) != artifact.Digest {
		return nil, artifacts.ErrArtifactNotFound
	}
	return append([]byte{}, value...), nil
}
func (f *fakeImportArtifacts) ReadRunBinding(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, bool, error) {
	if f.missingBinding {
		return auditstore.ExactArtifact{}, nil, false, artifacts.ErrArtifactNotFound
	}
	return f.runDescriptor, append([]byte{}, f.runPayload...), f.frozen, nil
}
func (*fakeImportArtifacts) ReadRunExact(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, error) {
	return auditstore.ExactArtifact{}, nil, errors.New("unexpected external evidence read")
}
func (f *fakeImportArtifacts) RetainRunExact(_ context.Context, _ string, source auditstore.ExactArtifact, _ string, target contracts.ArtifactRef) (auditstore.ExactArtifact, error) {
	revision := "retained-" + *source.Ref.Revision
	target.Revision = &revision
	source.Ref = target
	return source, nil
}
func (f *fakeImportArtifacts) PutImmutableProject(_ context.Context, _ string, target contracts.ArtifactRef, payload artifacts.Payload) (auditstore.ExactArtifact, error) {
	if f.writes == nil {
		return auditstore.ExactArtifact{}, errors.New("unexpected report write")
	}
	f.writes[target.Name] = append([]byte{}, payload.Data...)
	revision := "revision-" + target.Name
	target.Revision = &revision
	return auditstore.ExactArtifact{
		Ref: target, Digest: digestBytes(payload.Data), MediaType: payload.MediaType,
		SizeBytes: int64(len(payload.Data)),
	}, nil
}

func loadResultProfile(t *testing.T) config.ResolvedAuditProfile {
	return loadResultProfileWithFindingConfirmation(t, "disabled")
}

func loadResultProfileWithFindingConfirmation(
	t *testing.T,
	policy string,
) config.ResolvedAuditProfile {
	t.Helper()
	root := t.TempDir()
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies", "execution-configs",
		"agent-templates", "workflows", "audit-profiles", "skills",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
		"llm-gateways/test.yaml": `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata: {name: test-gateway, version: "1"}
spec: {protocol: openai-compatible@1, url: "http://127.0.0.1:4000/v1"}
`,
		"model-policies/worker.yaml": `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: worker, version: "1"}
spec: {model: worker-model, maxOutputTokens: 1024, maxModelCalls: 2, maxToolCalls: 4, maxTotalTokens: 4096, temperature: 0}
`,
		"agent-templates/worker.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: audit-worker, version: "1"}
spec:
  description: Produces one bounded result
  runtime: adk@1
  instructions: {ref: instructions/worker.md}
  modelPolicy: worker@1
  toolsets: [{ref: run-artifacts@1, tools: [read_artifact, write_artifact]}]
  sandboxProfile: local-workdir@1
`,
		"workflows/check.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: audit-check, version: "1"}
spec:
  parameters: {}
  inputs: {task: {required: true, mediaTypes: [application/zip]}}
  outputs: {result: {required: true, mediaTypes: [application/zip]}}
  executionConfig: {workers: {llmGateway: test-gateway@1, credential: development-worker}}
  entryStage: check
  stages:
    check:
      objective: Evaluate one checklist item
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents: {worker: {template: audit-worker@1}}
      context: {artifacts: {task: {namespace: inputs, name: task, required: true}}}
      result: {artifacts: {result: {required: true, mediaTypes: [application/zip], from: {namespace: worker, name: result}}}}
      workflowOutputs: {result: result}
      on: {succeeded: {succeed: {}}, failed: {fail: {}}, interrupted: {fail: {}}}
`,
		"audit-profiles/checklist.yaml": `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: test-checklist, version: "1"}
spec:
  mode: custom-checklist
  standards: []
  inputs: {checklist: {required: true, mediaTypes: [application/json]}}
  inventory: {implementation: checklist@1, sourceInput: checklist, itemWorkflowRole: check}
  workflows:
    check:
      ref: audit-check@1
      inputs: {task: {source: item-package}}
      parameters: {}
      outputs: {result: result}
  execution:
    {roundMode: fixed-barrier, maxRounds: 1, batchSize: 1, maxItemsPerRound: 10, maxItemsTotal: 10,
     maxSubmittedRuns: 20, maxItemRunAttempts: 2, deadlineSeconds: 3600, maxEvidenceBytes: 1048576,
     incompleteRound: assess-with-gaps}
  interaction: {activeChecks: prohibited, findingConfirmation: disabled, notApplicable: profile-rule, reportAcceptance: automatic}
`,
	}
	files["audit-profiles/checklist.yaml"] = strings.Replace(
		files["audit-profiles/checklist.yaml"],
		"findingConfirmation: disabled", "findingConfirmation: "+policy, 1,
	)
	for name, body := range files {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	return profile
}

func refKey(ref contracts.ArtifactRef) string {
	revision := ""
	if ref.Revision != nil {
		revision = *ref.Revision
	}
	return fmt.Sprintf("%s/%s@%s", ref.Namespace, ref.Name, revision)
}

func stringPointer(value string) *string { return &value }

func containsBytes(value []byte, fragment string) bool {
	return strings.Contains(string(value), fragment)
}
