package agentskills

import (
	"errors"
	"fmt"
)

const (
	CodeArchiveInvalid  = "skill_archive_invalid"
	CodePathInvalid     = "skill_path_invalid"
	CodeMemberForbidden = "skill_member_forbidden"
	CodeManifestInvalid = "skill_manifest_invalid"
	CodeNameMismatch    = "skill_name_mismatch"
	CodeLimitExceeded   = "skill_limit_exceeded"
)

// ValidationError is deliberately small: callers may safely expose Code and
// Member, but must not receive archive content, parser diagnostics, or host
// paths through Error.
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

func validationError(code, member string) error {
	return &ValidationError{Code: code, Member: member}
}

// ErrorCode returns a stable public classification for validation failures.
func ErrorCode(err error) string {
	var validation *ValidationError
	if errors.As(err, &validation) {
		return validation.Code
	}
	return ""
}
