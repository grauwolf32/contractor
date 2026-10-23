package contracts

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedIdempotencyKeyCases(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadIdempotencyKeyCases(t)
	if cases.MaxLength != MaxIdempotencyKeyLength {
		t.Fatalf("shared Idempotency-Key maxLength = %d, want %d", cases.MaxLength, MaxIdempotencyKeyLength)
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, value := range group.values {
			if ValidIdempotencyKey(value) != group.valid {
				t.Errorf("ValidIdempotencyKey(%q) = %v, want %v", value, !group.valid, group.valid)
			}
		}
	}
}
