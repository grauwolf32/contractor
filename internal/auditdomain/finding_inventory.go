package auditdomain

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strconv"
)

// BuildFindingInventory deterministically expands exact admitted proposals
// into one logical AuditItem per proposed check. The source document is itself
// immutable and exact, allowing the resulting task packages to share one
// bounded source identity while retaining each proposal revision separately.
func BuildFindingInventory(
	source []byte,
	options InventoryOptions,
) (Inventory, error) {
	document, err := DecodeFindingInventory(source)
	if err != nil {
		return Inventory{}, err
	}
	basisSubjects := make([]map[string]any, 0)
	subjects := make([]inventorySubject, 0)
	for _, proposal := range document.Proposals {
		limitations := copyStrings(proposal.Document.Limitations)
		sort.Strings(limitations)
		for _, ordinal := range proposal.SelectedCheckOrdinals {
			check := proposal.Document.ProposedChecks[ordinal]
			finding := &FindingTask{
				ReceiptID: proposal.ReceiptID, ProposalRef: copyArtifactRef(proposal.Proposal.Ref),
				ProposalDigest: proposal.Proposal.Digest, ProposedCheckOrdinal: ordinal,
				Objective: check.Objective, Method: check.Method, Limitations: copyStrings(limitations),
			}
			itemKey := findingCheckItemKey(proposal.ReceiptID, ordinal)
			basisSubjects = append(basisSubjects, map[string]any{
				"receipt_id":             proposal.ReceiptID,
				"proposal_ref":           proposal.Proposal.Ref,
				"proposal_digest":        proposal.Proposal.Digest,
				"proposed_check_ordinal": ordinal,
				"objective":              check.Objective,
				"method":                 check.Method,
				"subject_key":            proposal.Document.Subject.Key,
				"limitations":            limitations,
			})
			subjects = append(subjects, inventorySubject{
				itemKey: itemKey, kind: "finding-verification",
				subjectKey: proposal.Document.Subject.Key, finding: finding,
				requested: []string{check.Method}, gaps: copyStrings(limitations),
			})
		}
	}
	basis := inventoryBasis{
		Schema: InventoryBasisSchema, Kind: "finding-candidates",
		Subjects: basisSubjects, Gaps: []string{},
	}
	return finishInventory(source, JSONMediaType, basis, subjects, options)
}

func findingCheckItemKey(receiptID string, ordinal int) string {
	identity := []byte(ReceiptCheckIdentity(receiptID, ordinal))
	digest := sha256.Sum256(identity)
	return "finding-check-" + hex.EncodeToString(digest[:16])
}

// ReceiptCheckIdentity is a stable internal tuple key shared by deterministic
// inventory generation and durable proposal-to-item admission.
func ReceiptCheckIdentity(receiptID string, ordinal int) string {
	return receiptID + "\x00" + strconv.Itoa(ordinal)
}
