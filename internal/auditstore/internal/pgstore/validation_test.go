package pgstore

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestClosedStateAndTransitionValidation(t *testing.T) {
	t.Parallel()
	for _, state := range []auditstore.AuditState{
		auditstore.AuditDraft, auditstore.AuditActive, auditstore.AuditPaused, auditstore.AuditFinalizing,
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
