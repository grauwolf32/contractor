package randomid

import (
	"regexp"
	"testing"
)

func TestNewFormatsPrefixedHexIdentity(t *testing.T) {
	pattern := regexp.MustCompile(`^run_[0-9a-f]{32}$`)
	first, err := New("run_")
	if err != nil || !pattern.MatchString(first) {
		t.Fatalf("New = %q, %v", first, err)
	}
	second, err := New("run_")
	if err != nil || second == first {
		t.Fatalf("New repeated an identity: %q, %v", second, err)
	}
}
