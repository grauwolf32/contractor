package auditimport

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestImporterSharesOneCopyForEvidenceCitingOneRevision(t *testing.T) {
	harness := newImportHarness(t)
	access := configureSharedExternalEvidence(t, &harness)
	// The budget admits the result and the cited revision exactly once. A
	// second copy of the same revision would exceed it at the store gate.
	harness.snapshot.Audit.Limits.MaxEvidenceBytes =
		harness.artifacts.runDescriptor.SizeBytes + access.descriptor.SizeBytes

	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	collected := harness.store.collected
	if err != nil || !worked || collected.Disposition != auditstore.CollectionAccepted {
		t.Fatalf("shared evidence collection = (%t, %v, %+v)", worked, err, collected)
	}
	if targets := harness.artifacts.retainedTargets; len(targets) != 2 {
		t.Fatalf("retained copies = %+v, want the result and one evidence copy", targets)
	}
	var evidenceRefs []contracts.ArtifactRef
	for _, link := range collected.Retained {
		if link.LogicalKey != "result/execution-item-1" {
			evidenceRefs = append(evidenceRefs, link.Artifact.Ref)
		}
	}
	if len(evidenceRefs) != 2 || refKey(evidenceRefs[0]) != refKey(evidenceRefs[1]) {
		t.Fatalf("evidence links do not share one retained copy: %+v", evidenceRefs)
	}
	if charged := storeChargedBytes(collected.Retained); charged != harness.snapshot.Audit.Limits.MaxEvidenceBytes {
		t.Fatalf("store would charge %d bytes, importer admitted %d",
			charged, harness.snapshot.Audit.Limits.MaxEvidenceBytes)
	}
}

func TestImporterSettlesEvidenceBudgetExhaustedAtCommit(t *testing.T) {
	harness := newImportHarness(t)
	rejectAcceptedForBudget(harness.store)

	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	assertBudgetExhaustedReceipt(t, harness, worked, err)
	item := harness.store.collected.Items
	if len(item) != 1 || item[0].Retryable || item[0].Result != nil ||
		item[0].Coverage.Status != auditstore.CoverageInconclusive {
		t.Fatalf("budget-exhausted item = %+v", item)
	}
}

func TestImporterSettlesRoleEvidenceBudgetExhaustedAtCommit(t *testing.T) {
	harness := newImportHarness(t)
	configureDiscoveryRole(t, &harness)
	rejectAcceptedForBudget(harness.store)

	worked, err := harness.importer.Collect(
		context.Background(), harness.claim, harness.snapshot, harness.execution,
	)
	assertBudgetExhaustedReceipt(t, harness, worked, err)
	if len(harness.store.collected.Items) != 0 {
		t.Fatalf("role receipt has items: %+v", harness.store.collected.Items)
	}
}

// The collection gate admits execution-failed only for failed Runs, so a
// budget receipt for a recovered scan must follow the terminal outcome.
func TestScanRecoveryDispositionFollowsTerminalOutcome(t *testing.T) {
	for outcome, want := range map[auditstore.TerminalOutcome]auditstore.CollectionDisposition{
		auditstore.TerminalSucceeded:        auditstore.CollectionMissingOutput,
		auditstore.TerminalFailed:           auditstore.CollectionExecutionFailed,
		auditstore.TerminalSubmissionFailed: auditstore.CollectionExecutionFailed,
		auditstore.TerminalCancelled:        auditstore.CollectionExecutionCancelled,
	} {
		execution := auditstore.Execution{TerminalOutcome: &outcome}
		if got := scanRecoveryDisposition(execution); got != want {
			t.Errorf("scan recovery disposition for %s = %s, want %s", outcome, got, want)
		}
	}
}

// storeChargedBytes mirrors the collection transaction: each distinct
// retained revision is charged once, however many links cite it.
func storeChargedBytes(links []auditstore.ArtifactLink) int64 {
	seen := make(map[string]struct{}, len(links))
	var total int64
	for _, link := range links {
		key := refKey(link.Artifact.Ref)
		if _, charged := seen[key]; !charged {
			seen[key] = struct{}{}
			total += link.Artifact.SizeBytes
		}
	}
	return total
}

func rejectAcceptedForBudget(store *fakeImportStore) {
	store.collectErr = func(params auditstore.CollectParams) error {
		if params.Disposition == auditstore.CollectionAccepted {
			return auditstore.ErrEvidenceBudgetExhausted
		}
		return nil
	}
}

func assertBudgetExhaustedReceipt(t *testing.T, harness importHarness, worked bool, err error) {
	t.Helper()
	collected := harness.store.collected
	if err != nil || !worked || collected.Disposition != auditstore.CollectionInvalidResult ||
		collected.ErrorCode == nil || *collected.ErrorCode != "evidence-budget-exhausted" ||
		len(collected.Retained) != 0 || collected.SourceOutput == nil {
		t.Fatalf("budget-exhausted collection = (%t, %v, %+v)", worked, err, collected)
	}
}

// configureSharedExternalEvidence makes two evidence records of the one result
// cite the same exact Run artifact revision.
func configureSharedExternalEvidence(t *testing.T, harness *importHarness) *evidenceReadArtifacts {
	t.Helper()
	pkg, err := auditdomain.DecodeCheckResultPackage(harness.artifacts.runPayload)
	if err != nil {
		t.Fatal(err)
	}
	revision := "evidence-revision"
	ref := contracts.ArtifactRef{Namespace: "evidence", Name: "source", Revision: &revision}
	pkg.Evidence.Evidence = []auditdomain.Evidence{
		{ID: "ev-1", Kind: "source", Summary: "First citation of the source", Artifact: &ref},
		{ID: "ev-2", Kind: "source", Summary: "Second citation of the source", Artifact: &ref},
	}
	pkg.Results.Results[0].EvidenceIDs = []string{"ev-1", "ev-2"}
	evidence, err := auditdomain.EncodeEvidence(pkg.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	results, err := auditdomain.EncodeCheckResultSet(pkg.Results)
	if err != nil {
		t.Fatal(err)
	}
	payload, rebuilt, err := auditdomain.BuildPackage("shared-evidence-results", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{
		{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: results},
		{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: auditdomain.JSONMediaType, Data: evidence},
	})
	if err != nil {
		t.Fatal(err)
	}
	harness.artifacts.runPayload = payload
	harness.artifacts.runDescriptor.Digest = rebuilt.Digest
	harness.artifacts.runDescriptor.SizeBytes = int64(len(payload))
	access := &evidenceReadArtifacts{
		ArtifactAccess: harness.artifacts,
		descriptor: auditstore.ExactArtifact{
			Ref: ref, Digest: auditdomain.DigestBytes([]byte("evidence")), MediaType: "text/plain", SizeBytes: 8,
		},
	}
	harness.importer.artifacts = access
	return access
}

func configureDiscoveryRole(t *testing.T, harness *importHarness) {
	t.Helper()
	profile := loadResultProfileWithDiscovery(t)
	discovery := profile.Workflows["inventory"]
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Audit.Profile = auditstore.ProfileIdentity{
		Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
	}
	harness.snapshot.Audit.ProfileSnapshot = profileSnapshot
	harness.execution.Role = auditstore.ExecutionDiscovery
	harness.execution.WorkflowRole = "inventory"
	attempt := 1
	harness.execution.RoleAttempt = &attempt
	harness.store.members = nil
	runs := harness.importer.runs.(*fakeImportRuns)
	runs.run.WorkflowName = discovery.Workflow.Ref.Name
	runs.run.WorkflowVersion = discovery.Workflow.Ref.Version
	runs.run.WorkflowSnapshot, err = json.Marshal(discovery.Workflow)
	if err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Round = &auditstore.Round{RoundID: *harness.execution.RoundID, Ordinal: 1}
}
