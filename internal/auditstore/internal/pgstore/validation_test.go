package pgstore

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestClosedStateAndTransitionValidation(t *testing.T) {
	t.Parallel()
	for _, state := range []auditstore.AuditState{
		auditstore.AuditDraft, auditstore.AuditActive, auditstore.AuditWaitingReview, auditstore.AuditPaused, auditstore.AuditFinalizing,
		auditstore.AuditCancelling, auditstore.AuditCompleted, auditstore.AuditCancelled, auditstore.AuditFailed, auditstore.AuditDeleting,
	} {
		if !state.Valid() {
			t.Fatalf("known Audit state %q is invalid", state)
		}
	}
	if auditstore.AuditState("unknown").Valid() || auditstore.CollectionDisposition("other").Valid() {
		t.Fatal("unknown closed value accepted")
	}
	base := auditstore.TransitionParams{
		OwnerID: "owner", AuditID: "audit", ExpectedRevision: 2,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
		IdempotencyKey: "pause-1", RequestDigest: testDigest("1"),
	}
	if err := validateTransition(base); err != nil {
		t.Fatalf("valid owner transition: %v", err)
	}
	base.TargetState = auditstore.AuditCompleted
	if err := validateTransition(base); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("controller-only/invalid transition error = %v", err)
	}
	base.ExpectedState, base.TargetState = auditstore.AuditPaused, auditstore.AuditActive
	if err := validateTransition(base); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("plain paused-to-active transition error = %v, want Resume only", err)
	}
	base.ExpectedState, base.TargetState = auditstore.AuditDraft, auditstore.AuditDeleting
	if err := validateTransition(base); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("owner draft-to-deleting transition error = %v, want RequestDelete only", err)
	}
	if err := validateResume(auditstore.ResumeParams{
		OwnerID: "owner", AuditID: "audit", ExpectedRevision: 2,
		IdempotencyKey: "resume-1", RequestDigest: testDigest("2"),
	}); err != nil {
		t.Fatalf("valid Resume: %v", err)
	}
	trusted := auditstore.TrustedTransitionParams{
		AuditID: "audit", ExpectedRevision: 2,
		ExpectedState: auditstore.AuditWaitingReview, TargetState: auditstore.AuditActive,
	}
	if err := validateTrustedTransition(trusted); err != nil {
		t.Fatalf("valid trusted transition: %v", err)
	}
	trusted.ExpectedState = auditstore.AuditCompleted
	if err := validateTrustedTransition(trusted); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("disallowed trusted transition error = %v", err)
	}
}

func TestReportSummaryRequiresMarkdown(t *testing.T) {
	revision := "report-r1"
	link := func(key, media string) auditstore.ArtifactLink {
		return auditstore.ArtifactLink{LogicalKey: key, Artifact: auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "report", Revision: &revision},
			Digest: testDigest("1"), MediaType: media, SizeBytes: 16,
		}, SourceProvenance: json.RawMessage(`{}`)}
	}
	for _, media := range []string{"text/markdown", "text/plain", "text/html", "text/markdown; charset=utf-8", ""} {
		params := auditstore.CommitReportParams{
			Claim:                 auditstore.ControllerClaim{AuditID: "audit", HolderID: "holder", Epoch: 1},
			ExpectedAuditRevision: 1, RoundID: "round", ExpectedRoundRevision: 1,
			RequestDigest: testDigest("2"),
			Machine:       link(auditstore.ReportMachineLogicalKey, "application/json"),
			Summary:       link(auditstore.ReportSummaryLogicalKey, media),
		}
		for name, err := range map[string]error{
			"commit": validateCommitReport(params),
			"review": validateReportCandidateLinks(params.Machine, params.Summary),
		} {
			if media != "text/markdown" {
				if !errors.Is(err, auditstore.ErrInvalid) {
					t.Errorf("%s accepted unsupported %s: %v", name, media, err)
				}
			} else if err != nil {
				t.Errorf("%s rejected supported %s: %v", name, media, err)
			}
		}
	}
}

func TestMaterializationAndExecutionRejectDuplicateMembership(t *testing.T) {
	t.Parallel()
	artifact := testExact("tasks", "one", "r1")
	params := auditstore.MaterializeRoundParams{
		OwnerID: "owner", AuditID: "audit", ExpectedRevision: 1,
		RoundID: "round-1", RoundOrdinal: 1, Manifest: testExact("audit", "worklist", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[]}`), DeadlineAt: time.Now().Add(time.Hour),
		IdempotencyKey: "start-1", RequestDigest: testDigest("2"),
		Items: []auditstore.MaterializedItem{
			{ItemID: "item-1", ItemKey: "check-1", Ordinal: 0, Kind: "checklist", SubjectKey: "subject-1", Task: artifact, Origin: testOrigin("check-1"), WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: emptyCoverage()},
			{ItemID: "item-2", ItemKey: "check-1", Ordinal: 1, Kind: "checklist", SubjectKey: "subject-2", Task: artifact, Origin: testOrigin("check-1"), WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: emptyCoverage()},
		},
	}
	if err := validateMaterialize(params); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("duplicate item key error = %v", err)
	}

	attempt := 1
	roundID := "round-1"
	intent := auditstore.CreateExecutionIntentParams{
		Claim:       auditstore.ControllerClaim{AuditID: "audit", HolderID: "controller", Epoch: 1},
		ExecutionID: "execution-1", RoundID: &roundID, Role: auditstore.ExecutionCheck,
		WorkflowRole: "check", LaneClass: resolved.LaneClassModel,
		Manifest:      testExact("audit", "execution", "r1"),
		SubmissionKey: "submission-1", RequestDigest: testDigest("3"),
		Members: []auditstore.ExecutionMemberIntent{
			{ExecutionItemID: "member-1", ItemID: "item-1", BatchOrdinal: 0, ItemAttempt: attempt, Task: artifact},
			{ExecutionItemID: "member-2", ItemID: "item-1", BatchOrdinal: 1, ItemAttempt: attempt, Task: artifact},
		},
	}
	if err := validateExecutionIntent(intent); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("duplicate execution membership error = %v", err)
	}
}

func TestCollectionValidationKeepsAttemptAndLogicalSettlementDistinct(t *testing.T) {
	t.Parallel()
	params := auditstore.CollectParams{
		Claim:     auditstore.ControllerClaim{AuditID: "audit", HolderID: "controller", Epoch: 1},
		ReceiptID: "receipt-1", ExecutionID: "execution-1",
		Disposition: auditstore.CollectionMissingOutput, RequestDigest: testDigest("4"),
		Items: []auditstore.CollectionItem{{
			ExecutionItemID: "member-1", Disposition: auditstore.CollectionMissingOutput,
			Retryable: true, FinalDisposition: auditstore.FinalMissingOutput,
			Coverage: emptyCoverage(),
		}},
	}
	if err := ValidateCollect(params); err != nil {
		t.Fatalf("valid retryable collection: %v", err)
	}
	params.Items[0].FinalDisposition = auditstore.FinalAccepted
	if err := ValidateCollect(params); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("mismatched final disposition error = %v", err)
	}
	params.Items[0].FinalDisposition = auditstore.FinalInvalidResult
	params.Items[0].Disposition = auditstore.CollectionInvalidResult
	params.Disposition = auditstore.CollectionInvalidResult
	if err := ValidateCollect(params); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("invalid result without exact source output error = %v", err)
	}
	invalidSource := testExact("outputs", "invalid-result", "r1")
	params.SourceOutput = &invalidSource
	if err := ValidateCollect(params); err != nil {
		t.Fatalf("invalid result with exact source output: %v", err)
	}
	params.SourceOutput = nil
	params.Disposition = auditstore.CollectionContractInvalid
	params.Items[0].Disposition = auditstore.CollectionContractInvalid
	params.Items[0].Retryable = false
	if err := ValidateCollect(params); err != nil {
		t.Fatalf("collection contract failure without invented source output: %v", err)
	}
}

func TestCollectionValidationScopesFindingReceiptsToOneMember(t *testing.T) {
	t.Parallel()
	result := testExact("outputs", "result", "r1")
	proposal := testExact("audit-findings", "candidate", "r1")
	proposal.MediaType = "application/json"
	member := func(executionItemID string, associations ...auditstore.FindingAssociation) auditstore.CollectionItem {
		return auditstore.CollectionItem{
			ExecutionItemID: executionItemID, Disposition: auditstore.CollectionAccepted,
			FinalDisposition: auditstore.FinalAccepted, Result: &result, Coverage: emptyCoverage(),
			FindingAssociations: associations,
		}
	}
	association := func(assessmentID string) auditstore.FindingAssociation {
		return auditstore.FindingAssociation{
			AssessmentID: assessmentID, ReceiptID: "receipt-1",
			Proposal: proposal, SemanticAssessment: "supported",
		}
	}
	params := auditstore.CollectParams{
		Claim:     auditstore.ControllerClaim{AuditID: "audit", HolderID: "controller", Epoch: 1},
		ReceiptID: "receipt", ExecutionID: "execution-1", Disposition: auditstore.CollectionAccepted,
		SourceOutput: &result, RequestDigest: testDigest("4"),
		Items: []auditstore.CollectionItem{
			member("member-1", association("assessment-1")),
			member("member-2", association("assessment-2")),
		},
	}
	if err := ValidateCollect(params); err != nil {
		t.Fatalf("distinct members verifying one receipt: %v", err)
	}
	params.Items[1].FindingAssociations[0].AssessmentID = "assessment-1"
	if err := ValidateCollect(params); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("duplicate finding assessment error = %v", err)
	}
	params.Items = []auditstore.CollectionItem{member("member-1", association("assessment-1"), association("assessment-2"))}
	if err := ValidateCollect(params); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("one member associating a receipt twice error = %v", err)
	}
}

func TestExactArtifactValidationBoundsStoredReference(t *testing.T) {
	t.Parallel()
	artifact := testExact("tasks", "one", "r1")
	artifact.Ref.Name = strings.Repeat("n", auditstore.MaxArtifactRefBytes)
	if err := validateExactArtifact("task", artifact, false); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("oversized exact ArtifactRef error = %v", err)
	}
	artifact = testExact("tasks", "one", "r1")
	artifact.Ref.Name = string([]byte{0xff})
	if err := validateExactArtifact("task", artifact, false); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("non-UTF-8 exact ArtifactRef error = %v", err)
	}
}

func testExact(namespace, name, revision string) auditstore.ExactArtifact {
	return auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision},
		Digest: testDigest("a"), MediaType: "application/zip", SizeBytes: 1,
	}
}

func testDigest(character string) string { return "sha256:" + strings.Repeat(character, 64) }

func emptyCoverage() auditstore.Coverage {
	return auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}
}

func testOrigin(entryKey string) auditstore.ItemOrigin {
	source := testExact("inputs", "source", "source-r1")
	return auditstore.ItemOrigin{
		Schema: auditstore.ItemOriginSchema, SourceRef: &source.Ref,
		SourceContentDigest: testDigest("b"), SourceMediaType: "application/json",
		CanonicalInventoryDigest: testDigest("c"), EntryKey: entryKey, EntryVersion: "1",
	}
}
