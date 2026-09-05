package auditdomain

import (
	"crypto/sha256"
	"encoding/hex"
)

// ArtifactNamespace is the Server-reserved ProjectScope namespace containing
// one Audit's immutable task, manifest, result, evidence, and report bindings.
// Keeping this derivation in the domain package prevents lifecycle cleanup
// from guessing which protected bindings belong to an Audit.
func ArtifactNamespace(auditID string) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("contractor.audit.identity.v1\x00audit\x00"))
	_, _ = digest.Write([]byte(auditID))
	return "audit-" + hex.EncodeToString(digest.Sum(nil))
}
