package artifacttransfer

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestDurationAllowsMaximumPayloadAtMinimumThroughput(t *testing.T) {
	if got := Duration(artifacts.MaxPayloadSize); got != 69*time.Second {
		t.Fatalf("64 MiB transfer deadline = %s, want 69s", got)
	}
	if got := Duration(-1); got != Duration(artifacts.MaxPayloadSize) {
		t.Fatalf("unknown-length transfer deadline = %s", got)
	}
	if got := Duration(1); got != 6*time.Second {
		t.Fatalf("small transfer deadline = %s, want 6s", got)
	}
}
