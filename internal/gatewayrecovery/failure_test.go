package gatewayrecovery

import (
	"testing"
	"time"
)

func TestRecoveryDelayUsesConfiguredCap(t *testing.T) {
	policy := Policy{InitialDelay: 3 * time.Second, MaxDelay: 19 * time.Second, AutomaticWindow: time.Minute}
	for i, want := range []time.Duration{3, 6, 12, 19, 19} {
		if got := policy.failureDelay(int64(i)); got != want*time.Second {
			t.Fatalf("failure %d delay=%s", i, got)
		}
	}
}
