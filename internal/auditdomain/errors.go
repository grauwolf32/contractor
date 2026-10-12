// Package auditdomain defines the deterministic, persistence-independent Audit
// package and inventory contracts.
package auditdomain

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
)

const (
	CodeInvalid            = "audit_document_invalid"
	CodeSchemaUnsupported  = "audit_schema_unsupported"
	CodeLimitExceeded      = "audit_limit_exceeded"
	CodePackageInvalid     = "audit_package_invalid"
	CodePackagePathInvalid = "audit_package_path_invalid"
	CodeMemberForbidden    = "audit_package_member_forbidden"
	CodeDigestMismatch     = "audit_digest_mismatch"
	CodeReferenceInvalid   = "audit_reference_invalid"
	CodeRemoteReference    = "audit_remote_reference"
	CodeInventoryInvalid   = "audit_inventory_invalid"
	CodeResultSetInvalid   = "audit_result_set_invalid"
)

// ValidationError exposes only a stable classification and a bounded field
// name. Parser diagnostics and caller-controlled document contents stay out of
// public errors, logs, and metrics.
type ValidationError struct {
	Code  string
	Field string
}

func (e *ValidationError) Error() string {
	if e.Field == "" {
		return e.Code
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Field)
}

// ErrorKind answers the error with its registered code. Every Code constant
// is registered; an unknown code is answered as audit_document_invalid.
func (e *ValidationError) ErrorKind() apierror.Kind {
	if kind, ok := apierror.Lookup(e.Code); ok {
		return kind
	}
	return apierror.AuditDocumentInvalid
}

func invalid(code, field string) error {
	if len(field) > MaximumDiagnosticBytes {
		field = field[:MaximumDiagnosticBytes]
	}
	return &ValidationError{Code: code, Field: field}
}

// ErrorCode returns the stable public classification for a validation error.
func ErrorCode(err error) string {
	var validation *ValidationError
	if errors.As(err, &validation) {
		return validation.Code
	}
	return ""
}
