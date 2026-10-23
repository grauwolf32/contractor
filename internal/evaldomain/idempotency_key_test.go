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
