package auditstore

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestClosedStateAndTransitionValidation(t *testing.T) {
	t.Parallel()
	for _, state := range []AuditState{
		AuditDraft, AuditActive, AuditWaitingReview, AuditPaused, AuditFinalizing,
		AuditCancelling, AuditCompleted, AuditCancelled, AuditFailed, AuditDeleting,
	} {
		if !state.Valid() {
			t.Fatalf("known Audit state %q is invalid", state)
		}
	}
	if AuditState("unknown").Valid() || CollectionDisposition("other").Valid() {
		t.Fatal("unknown closed value accepted")
	}
	base := TransitionParams{
		OwnerID: "owner", AuditID: "audit", ExpectedRevision: 2,
		ExpectedState: AuditActive, TargetState: AuditPaused,
		IdempotencyKey: "pause-1", RequestDigest: testDigest("1"),
	}
	if err := validateTransition(base); err != nil {
		t.Fatalf("valid owner transition: %v", err)
	}
	base.TargetState = AuditCompleted
	if err := validateTransition(base); !errors.Is(err, ErrInvalid) {
		t.Fatalf("controller-only/invalid transition error = %v", err)
	}
}

func TestReportSummaryMediaTypes(t *testing.T) {
	revision := "report-r1"
	link := func(key, media string) ArtifactLink {
		return ArtifactLink{LogicalKey: key, Artifact: ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "report", Revision: &revision},
			Digest: testDigest("1"), MediaType: media, SizeBytes: 16,
		}, SourceProvenance: json.RawMessage(`{}`)}
	}
	for _, media := range []string{"text/markdown", "text/plain", "text/html"} {
		params := CommitReportParams{
			Claim:                 ControllerClaim{AuditID: "audit", HolderID: "holder", Epoch: 1},
			ExpectedAuditRevision: 1, RoundID: "round", ExpectedRoundRevision: 1,
			RequestDigest: testDigest("2"),
			Machine:       link(ReportMachineLogicalKey, "application/json"),
			Summary:       link(ReportSummaryLogicalKey, media),
		}
		for name, err := range map[string]error{
			"commit": validateCommitReport(params),
			"review": validateReportCandidateLinks(params.Machine, params.Summary),
		} {
			if media == "text/html" {
				if !errors.Is(err, ErrInvalid) {
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
	params := MaterializeRoundParams{
		OwnerID: "owner", AuditID: "audit", ExpectedRevision: 1,
		RoundID: "round-1", RoundOrdinal: 1, Manifest: testExact("audit", "worklist", "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":[]}`), DeadlineAt: time.Now().Add(time.Hour),
		IdempotencyKey: "start-1", RequestDigest: testDigest("2"),
		Items: []MaterializedItem{
			{ItemID: "item-1", ItemKey: "check-1", Ordinal: 0, Kind: "checklist", SubjectKey: "subject-1", Task: artifact, Origin: testOrigin("check-1"), WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage()},
			{ItemID: "item-2", ItemKey: "check-1", Ordinal: 1, Kind: "checklist", SubjectKey: "subject-2", Task: artifact, Origin: testOrigin("check-1"), WorkflowRole: "check", InitialState: ItemReady, Coverage: emptyCoverage()},
		},
	}
	if err := validateMaterialize(params); !errors.Is(err, ErrInvalid) {
		t.Fatalf("duplicate item key error = %v", err)
	}

	attempt := 1
	roundID := "round-1"
	intent := CreateExecutionIntentParams{
		Claim:       ControllerClaim{AuditID: "audit", HolderID: "controller", Epoch: 1},
		ExecutionID: "execution-1", RoundID: &roundID, Role: ExecutionCheck,
		WorkflowRole:  "check",
		Manifest:      testExact("audit", "execution", "r1"),
		SubmissionKey: "submission-1", RequestDigest: testDigest("3"),
		Members: []ExecutionMemberIntent{
			{ExecutionItemID: "member-1", ItemID: "item-1", BatchOrdinal: 0, ItemAttempt: attempt, Task: artifact},
			{ExecutionItemID: "member-2", ItemID: "item-1", BatchOrdinal: 1, ItemAttempt: attempt, Task: artifact},
		},
	}
	if err := validateExecutionIntent(intent); !errors.Is(err, ErrInvalid) {
		t.Fatalf("duplicate execution membership error = %v", err)
	}
}

func TestCollectionValidationKeepsAttemptAndLogicalSettlementDistinct(t *testing.T) {
	t.Parallel()
	params := CollectParams{
		Claim:     ControllerClaim{AuditID: "audit", HolderID: "controller", Epoch: 1},
		ReceiptID: "receipt-1", ExecutionID: "execution-1",
		Disposition: CollectionMissingOutput, RequestDigest: testDigest("4"),
		Items: []CollectionItem{{
			ExecutionItemID: "member-1", Disposition: CollectionMissingOutput,
			Retryable: true, FinalDisposition: FinalMissingOutput,
			Coverage: emptyCoverage(),
		}},
	}
	if err := validateCollect(params); err != nil {
		t.Fatalf("valid retryable collection: %v", err)
	}
	params.Items[0].FinalDisposition = FinalAccepted
	if err := validateCollect(params); !errors.Is(err, ErrInvalid) {
		t.Fatalf("mismatched final disposition error = %v", err)
	}
	params.Items[0].FinalDisposition = FinalInvalidResult
	params.Items[0].Disposition = CollectionInvalidResult
	params.Disposition = CollectionInvalidResult
	if err := validateCollect(params); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid result without exact source output error = %v", err)
	}
	invalidSource := testExact("outputs", "invalid-result", "r1")
	params.SourceOutput = &invalidSource
	if err := validateCollect(params); err != nil {
		t.Fatalf("invalid result with exact source output: %v", err)
	}
	params.SourceOutput = nil
	params.Disposition = CollectionContractInvalid
	params.Items[0].Disposition = CollectionContractInvalid
	params.Items[0].Retryable = false
	if err := validateCollect(params); err != nil {
		t.Fatalf("collection contract failure without invented source output: %v", err)
	}
}

func TestExactArtifactValidationBoundsStoredReference(t *testing.T) {
	t.Parallel()
	artifact := testExact("tasks", "one", "r1")
	artifact.Ref.Name = strings.Repeat("n", MaxArtifactRefBytes)
	if err := validateExactArtifact("task", artifact, false); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized exact ArtifactRef error = %v", err)
	}
	artifact = testExact("tasks", "one", "r1")
	artifact.Ref.Name = string([]byte{0xff})
	if err := validateExactArtifact("task", artifact, false); !errors.Is(err, ErrInvalid) {
		t.Fatalf("non-UTF-8 exact ArtifactRef error = %v", err)
	}
}

func testExact(namespace, name, revision string) ExactArtifact {
	return ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision},
		Digest: testDigest("a"), MediaType: "application/zip", SizeBytes: 1,
	}
}

func testDigest(character string) string { return "sha256:" + strings.Repeat(character, 64) }

func emptyCoverage() Coverage {
	return Coverage{Status: CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}
}

func testOrigin(entryKey string) ItemOrigin {
	source := testExact("inputs", "source", "source-r1")
	return ItemOrigin{
		Schema: ItemOriginSchema, SourceRef: &source.Ref,
		SourceContentDigest: testDigest("b"), SourceMediaType: "application/json",
		CanonicalInventoryDigest: testDigest("c"), EntryKey: entryKey, EntryVersion: "1",
	}
}
