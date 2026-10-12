package auditprofile

import "github.com/grauwolf32/contractor/internal/config/resolved"

// ProfileDigest is the digest profile pins under its own identity. Tests use
// it to build a snapshot whose digest is valid while its content is not, so a
// rejection proves the decoder checked the content.
func ProfileDigest(profile ResolvedAuditProfile) (string, error) {
	return auditProfileDigest(resolved.Selector{ID: profile.Ref.Name, Version: profile.Ref.Version}, profile)
}
