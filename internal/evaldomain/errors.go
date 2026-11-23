package evaldomain

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/apierror"
)

// IsCode tests a domain error without classifying unrelated infrastructure
// failures as a recoverable evaluation condition.
func IsCode(err error, code string) bool {
	var domainError *Error
	return errors.As(err, &domainError) && domainError.Code == code
}

// Error is safe to expose. Neither decoder messages nor submitted values are
// copied into errors, including credentials, private expected data and refs.
// Its code, status and message are the code's public error registry entry
// (apierror); Recovery tells the client what to do instead of retrying.
type Error struct {
	Code     string `json:"code"`
	Message  string `json:"message"`
	Recovery string `json:"recovery"`
	Status   int    `json:"-"`
}

func (e *Error) Error() string { return e.Message }

// ErrorKind answers the error with its registered code; an unregistered code
// is answered as eval_invalid.
func (e *Error) ErrorKind() apierror.Kind {
	if kind, ok := apierror.Lookup(e.Code); ok {
		return kind
	}
	return apierror.EvalInvalid
}

// PublicMessage is the error's own safe message.
func (e *Error) PublicMessage() string { return e.Message }

// recoveries maps every Eval code to the recovery action it reports.
var recoveries = map[string]string{
	"eval_invalid":                 "edit_draft",
	"eval_limit_exceeded":          "edit_draft",
	"eval_not_found":               "none",
	"eval_precondition_required":   "reload",
	"eval_revision_mismatch":       "reload",
	"eval_idempotency_conflict":    "retry_same_request",
	"eval_not_ready":               "edit_draft",
	"eval_preparation_unavailable": "wait",
	"eval_pin_mismatch":            "duplicate",
	"eval_external_control":        "none",
	"eval_view_changed":            "reload",
	"eval_member_conflict":         "inspect_execution",
	"eval_evidence_unavailable":    "inspect_execution",
	"eval_producer_stale":          "inspect_execution",
	"eval_budget_exhausted":        "duplicate",
	"eval_project_deleting":        "wait",
}

// Failure returns the Eval error of code with its registered status and
// message; an unknown code is eval_invalid.
func Failure(code string) *Error {
	recovery, known := recoveries[code]
	kind, registered := apierror.Lookup(code)
	if !known || !registered {
		return Failure("eval_invalid")
	}
	return &Error{Code: code, Message: kind.Message(), Recovery: recovery, Status: kind.Status()}
}
