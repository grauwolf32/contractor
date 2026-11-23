package contracts_test

import (
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// A SecretString redacts itself for every fmt verb, flag and nesting, for
// log/slog and for encoding/json (SEC-R88); before V414-011 %d printed
// "{%!d(string=...)}".
func TestSecretStringRedactsForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	secret := contracts.NewSecretString("hunter2-plaintext")
	contractstest.AssertRedacted(t, secret, "hunter2-plaintext", "[REDACTED]")
	if got := fmt.Sprintf("%#v", secret); got != "contracts.SecretString([REDACTED])" {
		t.Fatalf("%%#v = %q", got)
	}
}
