package credentials

import (
	"errors"
	"testing"
)

// Operation request digests are stored for idempotent replay, so their
// encoding must not change.
func TestOperationRequestDigestIsPinned(t *testing.T) {
	digest, err := operationRequestDigest(struct {
		CredentialID string `json:"credentialId"`
		ActorID      string `json:"actorId"`
	}{CredentialID: "cred-1", ActorID: "actor-1"})
	if err != nil || digest != "sha256:ba3ac17410007b2df2d0a40dd7a70fc64c9b818126bbc4dba9159b6a8ec769d3" {
		t.Fatalf("operationRequestDigest() = %q, %v", digest, err)
	}
	if _, err := operationRequestDigest(func() {}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("operationRequestDigest(unencodable) error = %v", err)
	}
}
