// Package auditservice owns public-facing Audit draft/start orchestration. It
// resolves mutable dependencies only before the immutable baseline commits;
// later controller work consumes the retained snapshot.
package auditservice

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var (
	ErrInvalid         = apierror.New(apierror.InvalidRequest, "audit service invalid request")
	ErrProfileNotFound = apierror.New(apierror.NotFound, "AuditProfile was not found")
	ErrUnsupported     = errors.New("AuditProfile is unsupported by this Server")
)

type CompatibilityReason string

const (
	ReasonEngagementCapabilityUnsupported   CompatibilityReason = "engagement_capability_unsupported"
	ReasonEngagementActiveChecksUnsupported CompatibilityReason = "engagement_active_checks_unsupported"
)

type UnsupportedError struct {
	Reasons []CompatibilityReason
}

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("%s: %v", ErrUnsupported, e.Reasons)
}

func (e *UnsupportedError) Unwrap() error { return ErrUnsupported }

// ErrorKind answers the error with audit_profile_unsupported; its reasons
// are the details.
func (e *UnsupportedError) ErrorKind() apierror.Kind { return apierror.AuditProfileUnsupported }

func unsupported(reasons []CompatibilityReason) error {
	return &UnsupportedError{Reasons: append([]CompatibilityReason(nil), reasons...)}
}
