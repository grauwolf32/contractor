package evaldomain

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedEvalMutationKeysFollowTheIdempotencyKeyGrammar binds a Managed
// Eval mutation only to a key in the public Idempotency-Key grammar; any
// other key, such as a!b or a/b, is eval_invalid.
func TestSharedEvalMutationKeysFollowTheIdempotencyKeyGrammar(t *testing.T) {
	t.Parallel()
	body := []byte(`{"kind":"prepare"}`)
	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, key := range cases.Valid {
		if identity, err := IdentifyMutation(key, `"1"`, true, "Command", body); err != nil || identity.Key != key {
			t.Errorf("valid key %q: identity %+v, error %v", key, identity, err)
		}
	}
	for _, key := range cases.Invalid {
		var failure *Error
		if _, err := IdentifyMutation(key, `"1"`, true, "Command", body); !errors.As(err, &failure) || failure.Code != "eval_invalid" {
			t.Errorf("invalid key %q: error %v, want eval_invalid", key, err)
		}
	}
}

// TestSharedEvalPreconditionsUseTheCounterReader reads a Managed Eval If-Match
// with the one counter precondition reader of the public API: the shared
// valid cases carry their revision, every other value, "01" included, is
// eval_invalid, and a missing required If-Match is eval_precondition_required.
func TestSharedEvalPreconditionsUseTheCounterReader(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadRevisionETagCases(t)
	for _, valid := range cases.Valid {
		if revision, err := MutationPrecondition("eval-key", valid.Header, true); err != nil || revision == nil || *revision != valid.Revision {
			t.Errorf("If-Match %q: revision %v, error %v; want %d", valid.Header, revision, err, valid.Revision)
		}
	}
	for _, raw := range cases.Rejected() {
		if raw == "" {
			continue // an absent If-Match is the missing precondition below
		}
		var failure *Error
		if _, err := MutationPrecondition("eval-key", raw, true); !errors.As(err, &failure) || failure.Code != "eval_invalid" {
			t.Errorf("If-Match %q: error %v, want eval_invalid", raw, err)
		}
	}
	var failure *Error
	if _, err := MutationPrecondition("eval-key", "", true); !errors.As(err, &failure) || failure.Code != "eval_precondition_required" {
		t.Errorf("missing If-Match: error %v, want eval_precondition_required", err)
	}
	if revision, err := MutationPrecondition("eval-key", "", false); err != nil || revision != nil {
		t.Errorf("optional absent If-Match: revision %v, error %v", revision, err)
	}
}
