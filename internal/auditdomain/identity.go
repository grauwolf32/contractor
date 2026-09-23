package auditdomain

import (
	"crypto/sha256"
	"encoding/hex"
)

// DigestBytes returns the "sha256:<hex>" content digest used for exact Audit
// artifacts, documents, and request identities.
func DigestBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// DeterministicID derives a stable "<prefix>-<hex>" identity from the Audit
// identity domain, the prefix, and each value separated by NUL bytes. The
// encoding is persisted in Audit, round, item, and finding identifiers and
// must never change.
func DeterministicID(prefix string, values ...string) string {
	digest := sha256.New()
	_, _ = digest.Write([]byte("contractor.audit.identity.v1\x00" + prefix))
	for _, value := range values {
		_, _ = digest.Write([]byte{0})
		_, _ = digest.Write([]byte(value))
	}
	return prefix + "-" + hex.EncodeToString(digest.Sum(nil))
}

// ArtifactNamespace is the Server-reserved ProjectScope namespace containing
// one Audit's immutable task, manifest, result, evidence, and report bindings.
// Keeping this derivation in the domain package prevents lifecycle cleanup
// from guessing which protected bindings belong to an Audit.
func ArtifactNamespace(auditID string) string {
	return DeterministicID("audit", auditID)
}
