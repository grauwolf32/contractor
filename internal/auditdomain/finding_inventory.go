package auditdomain

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
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
		checks := newFindingChecks(proposal)
		for _, ordinal := range proposal.SelectedCheckOrdinals {
			basisSubject, subject := checks.subject(ordinal)
			basisSubjects = append(basisSubjects, basisSubject)
			subjects = append(subjects, subject)
		}
	}
	basis := newFindingInventoryBasis(basisSubjects)
	return finishInventory(source, JSONMediaType, basis, subjects, options)
}

// findingChecks derives the per-check inventory entries of one admitted
// proposal. BuildFindingInventory and FindingRoundBudget share it, so a Round
// is measured with exactly the entries it is later built from.
type findingChecks struct {
	proposal    FindingInventoryProposal
	subjectKey  string
	limitations []string
}

func newFindingChecks(proposal FindingInventoryProposal) findingChecks {
	// With an unspecified affected subject, verification addresses the
	// retained finding receipt itself; the proposal remains subject:null.
	subjectKey := proposal.ReceiptID
	if proposal.Document.Subject != nil {
		subjectKey = proposal.Document.Subject.Key
	}
	limitations := slices.Clone(proposal.Document.Limitations)
	sort.Strings(limitations)
	return findingChecks{proposal: proposal, subjectKey: subjectKey, limitations: limitations}
}

// subject returns the canonical inventory entry and the item subject of one
// proposed check. The ordinal must address one of the proposal's checks.
func (checks findingChecks) subject(ordinal int) (map[string]any, inventorySubject) {
	proposal := checks.proposal
	check := proposal.Document.ProposedChecks[ordinal]
	finding := &FindingTask{
		ReceiptID: proposal.ReceiptID, ProposalRef: proposal.Proposal.Ref.Clone(),
		ProposalDigest: proposal.Proposal.Digest, ProposedCheckOrdinal: ordinal,
		Objective: check.Objective, Method: check.Method, Limitations: slices.Clone(checks.limitations),
	}
	basisSubject := map[string]any{
		"receipt_id":             proposal.ReceiptID,
		"proposal_ref":           proposal.Proposal.Ref,
		"proposal_digest":        proposal.Proposal.Digest,
		"proposed_check_ordinal": ordinal,
		"objective":              check.Objective,
		"method":                 check.Method,
		"subject_key":            checks.subjectKey,
		"limitations":            checks.limitations,
	}
	return basisSubject, inventorySubject{
		itemKey: findingCheckItemKey(proposal.ReceiptID, ordinal), kind: "finding-verification",
		subjectKey: checks.subjectKey, finding: finding,
		requested: []string{check.Method}, gaps: slices.Clone(checks.limitations),
	}
}

func newFindingInventoryBasis(subjects []map[string]any) inventoryBasis {
	return inventoryBasis{
		Schema: InventoryBasisSchema, Kind: "finding-candidates",
		Subjects: subjects, Gaps: []string{},
	}
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
