package auditstore

import (
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// MaxFindingActorBytes bounds the authoritative actor stored on a Project Finding.
const MaxFindingActorBytes = 256

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
		f.ActorID == nil || !ValidFindingIdentity(*f.ActorID, MaxFindingActorBytes) || f.AssessedAt == nil || f.AssessedAt.IsZero() {
		return ErrInvalid
	}
	if f.AnalystSeverity != nil && !f.AnalystSeverity.Valid() {
		return ErrInvalid
	}
	if f.AssessmentShape() != nil || f.DuplicateTargetID != nil && (!contracts.ValidResourceID(*f.DuplicateTargetID) || *f.DuplicateTargetID == f.FindingID) {
		return ErrInvalid
	}

	return nil
}

func validFindingText(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) && !strings.ContainsRune(value, 0) && len(value) <= maximum
}

// AssessmentShape is shared by full and compact readers of canonical state.
func (f ProjectFinding) AssessmentShape() error {
	var state *string
	if f.State != nil {
		value := string(*f.State)
		state = &value
	}
	verdict, severity, duplicate := "", "", ""
	if f.AnalystVerdict != nil {
		verdict = string(*f.AnalystVerdict)
	}
	if f.AnalystSeverity != nil {
		severity = string(*f.AnalystSeverity)
	}
	if f.DuplicateTargetID != nil {
		duplicate = *f.DuplicateTargetID
	}
	return auditdomain.ValidateFindingAssessment(state, verdict, severity, duplicate)
}
