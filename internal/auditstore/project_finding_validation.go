package auditstore

import (
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Validate checks the canonical public state, including mutation-receipt reads.
// Source ratings never participate in these invariants.
func (f ProjectFinding) Validate() error {
	if !contracts.ValidResourceID(f.FindingID) || !contracts.ValidResourceID(f.ProjectID) ||
		!contracts.ValidResourceID(f.FirstReceiptID) || f.FirstProposal.Validate() != nil || !f.FirstProposal.HasMetadata() ||
		!validFindingText(f.Title, 65536) || f.Revision < 1 || f.Revision > contracts.MaxExactInteger ||
		f.CreatedAt.IsZero() || f.UpdatedAt.IsZero() || f.UpdatedAt.Before(f.CreatedAt) {
		return ErrInvalid
	}
	if f.State == nil {
		if f.AnalystVerdict != nil || f.AnalystSeverity != nil || f.Rationale != nil || f.ActorID != nil || f.AssessedAt != nil || f.DuplicateTargetID != nil {
			return ErrInvalid
		}
		return nil
	}
	if !f.State.Valid() || f.AnalystVerdict == nil || !f.AnalystVerdict.Valid() || f.Rationale == nil || !validFindingText(*f.Rationale, 65536) ||
		f.ActorID == nil || !ValidReviewIdentity(*f.ActorID,256) || f.AssessedAt == nil || f.AssessedAt.IsZero() {
		return ErrInvalid
	}
	states := map[AnalystVerdict]FindingState{VerdictTruePositive: FindingConfirmed, VerdictFalsePositive: FindingRejected, VerdictDuplicate: FindingDuplicate, VerdictReopen: FindingProposed, VerdictNeedsEvidence: FindingNeedsEvidence}
	if states[*f.AnalystVerdict] != *f.State || (*f.AnalystVerdict == VerdictTruePositive) != (f.AnalystSeverity != nil) ||
		(f.AnalystSeverity != nil && !f.AnalystSeverity.Valid()) || (*f.AnalystVerdict == VerdictDuplicate) != (f.DuplicateTargetID != nil) ||
		(f.DuplicateTargetID != nil && (!contracts.ValidResourceID(*f.DuplicateTargetID) || *f.DuplicateTargetID == f.FindingID)) {
		return ErrInvalid
	}
	return nil
}

func validFindingText(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) && !strings.ContainsRune(value, 0) && len(value) <= maximum
}
