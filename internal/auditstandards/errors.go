package auditstandards

import (
	"errors"
	"fmt"
)

var (
	ErrNotFound = errors.New("Audit standard package not found")
	ErrInvalid  = errors.New("invalid Audit standard package")
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
	CodeCatalogDrift     = "audit_standard_catalog_drift"
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

func ErrorCode(err error) string {
	var validation *ValidationError
	if errors.As(err, &validation) {
		return validation.Code
	}
	if errors.Is(err, ErrDrift) {
		return CodeCatalogDrift
	}
	return ""
}
