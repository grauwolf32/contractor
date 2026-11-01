package auditdomain

import (
	"crypto/sha256"
	"encoding/base64"
)

const FindingSourceHistorySchema = "contractor.finding.source-history.v1"

// FindingSourceHistoryPrefix selects only one receipt's ordinary immutable
// history artifacts. It keeps resource IDs out of the Artifact name grammar.
func FindingSourceHistoryPrefix(receiptID string) string {
	digest := sha256.Sum256([]byte(receiptID))
	return "s." + base64.RawURLEncoding.EncodeToString(digest[:]) + "."
}

// FindingSourceHistoryName reuses a source operation's existing stable ID.
// One source delivery never gains another append identity.
func FindingSourceHistoryName(receiptID, kind, recordID string) string {
	digest := sha256.Sum256([]byte(recordID))
	return FindingSourceHistoryPrefix(receiptID) + kind + "." + base64.RawURLEncoding.EncodeToString(digest[:])
}
