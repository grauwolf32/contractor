package runtimeconfig

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedRuntimeManagementKeysFollowTheIdempotencyKeyGrammar keeps
// RuntimeConfig publication, label bindings and Agent label commands to the
// public Idempotency-Key grammar before a key is digested into a receipt.
func TestSharedRuntimeManagementKeysFollowTheIdempotencyKeyGrammar(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, key := range cases.Valid {
		if digest, err := DigestIdempotencyKey(key); err != nil || digest == "" {
			t.Errorf("valid key %q: digest %q, error %v", key, digest, err)
		}
	}
	for _, key := range cases.Invalid {
		if _, err := DigestIdempotencyKey(key); !errors.Is(err, ErrInvalid) {
			t.Errorf("invalid key %q: error %v, want ErrInvalid", key, err)
		}
	}
}
