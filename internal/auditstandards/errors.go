package auditstandards

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
)

var (
	ErrNotFound = apierror.New(apierror.NotFound, "Audit standard package not found")
	ErrInvalid  = apierror.New(apierror.InvalidRequest, "invalid Audit standard package")
	ErrDrift    = errors.New("Audit standard package drift")
)

const (
	CodeArchiveInvalid   = "audit_standard_archive_invalid"
	CodePathInvalid      = "audit_standard_path_invalid"
	CodeMemberForbidden  = "audit_standard_member_forbidden"
	CodeManifestInvalid  = "audit_standard_manifest_invalid"
	CodeIdentityMismatch = "audit_standard_identity_mismatch"
	CodeLimitExceeded    = "audit_standard_limit_exceeded"
	CodeLicenseInvalid   = "audit_standard_license_invalid"
	CodeDanglingMapping  = "audit_standard_dangling_mapping"
)

type ValidationError struct {
	Code   string
	Member string
}

func (e *ValidationError) Error() string {
	if e.Member != "" {
		return fmt.Sprintf("%s: member %q", e.Code, e.Member)
	}
	return e.Code
}

func (e *ValidationError) Unwrap() error { return ErrInvalid }

func validationError(code, member string) error {
	return &ValidationError{Code: code, Member: member}
}
