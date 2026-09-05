// Package auditservice owns public-facing Audit draft/start orchestration. It
// resolves mutable dependencies only before the immutable baseline commits;
// later controller work consumes the retained snapshot.
package auditservice

import (
	"errors"
	"fmt"
)

var (
	ErrInvalid         = errors.New("audit service invalid request")
	ErrProfileNotFound = errors.New("AuditProfile was not found")
	ErrUnsupported     = errors.New("AuditProfile is unsupported by this Server")
)

type CompatibilityReason string

const (
	ReasonDiscoveryUnsupported             CompatibilityReason = "discovery_unsupported"
	ReasonAssessmentUnsupported            CompatibilityReason = "assessment_unsupported"
	ReasonMultipleRoundsUnsupported        CompatibilityReason = "multiple_rounds_unsupported"
	ReasonBatchingUnsupported              CompatibilityReason = "batching_unsupported"
	ReasonAutomaticActiveChecksUnsupported CompatibilityReason = "automatic_active_checks_unsupported"
	ReasonActiveCheckApprovalUnsupported   CompatibilityReason = "active_check_approval_unsupported"
	ReasonFindingConfirmationUnsupported   CompatibilityReason = "finding_confirmation_unsupported"
	ReasonManualApplicabilityUnsupported   CompatibilityReason = "manual_applicability_unsupported"
	ReasonReportAcceptanceUnsupported      CompatibilityReason = "report_acceptance_unsupported"
	ReasonManualItemUnsupported            CompatibilityReason = "manual_item_unsupported"
)

type UnsupportedError struct {
	Reasons []CompatibilityReason
}

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("%s: %v", ErrUnsupported, e.Reasons)
}

func (e *UnsupportedError) Unwrap() error { return ErrUnsupported }

func unsupported(reasons []CompatibilityReason) error {
	return &UnsupportedError{Reasons: append([]CompatibilityReason(nil), reasons...)}
}
