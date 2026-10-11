package auditservice

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
)

// TestActionReviewDecisionUsesTheIdempotencyKeyGrammar covers V368-023: the
// action decision accepts exactly the stored Idempotency-Key grammar, so a key
// the database would refuse is ErrInvalid before any storage access.
func TestActionReviewDecisionUsesTheIdempotencyKeyGrammar(t *testing.T) {
	t.Parallel()
	valid := DecideActionReviewParams{
		OwnerID: "owner", AuditID: "audit", RequestID: "request", DecisionID: "decision",
		ExpectedRequestRevision: 1, Action: ReviewApprove, Rationale: "The exact action is approved.",
		IdempotencyKey: "decision-key_1.2:3", RequestDigest: contentdigest.Bytes([]byte("decision")),
	}
	if err := validateActionDecision(valid); err != nil {
		t.Fatalf("valid action decision = %v", err)
	}
	for _, key := range []string{
		"", "key with spaces", "-leading", ".leading", "key/slash", "käse", strings.Repeat("a", 129),
	} {
		params := valid
		params.IdempotencyKey = key
		// A zero Service has no pool: reaching storage would panic.
		if _, err := (&Service{}).DecideActionReview(context.Background(), params); !errors.Is(err, auditstore.ErrInvalid) {
			t.Fatalf("action decision with key %q = %v, want ErrInvalid", key, err)
		}
	}
	longest := valid
	longest.IdempotencyKey = strings.Repeat("a", 128)
	if err := validateActionDecision(longest); err != nil {
		t.Fatalf("128-character key = %v", err)
	}
}
