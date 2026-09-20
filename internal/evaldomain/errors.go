package evaldomain

import "net/http"

// Error is safe to expose. Neither decoder messages nor submitted values are
// copied into errors, including credentials, private expected data and refs.
type Error struct {
	Code     string `json:"code"`
	Message  string `json:"message"`
	Recovery string `json:"recovery"`
	Status   int    `json:"-"`
}

func (e *Error) Error() string { return e.Message }

func Failure(code string) *Error {
	e := &Error{Code: code, Status: http.StatusConflict, Recovery: "reload"}
	switch code {
	case "eval_invalid":
		e.Status, e.Message, e.Recovery = http.StatusUnprocessableEntity, "Invalid evaluation document.", "edit_draft"
	case "eval_limit_exceeded":
		e.Status, e.Message, e.Recovery = http.StatusUnprocessableEntity, "Evaluation document exceeds format bounds.", "edit_draft"
	case "eval_not_found":
		e.Status, e.Message, e.Recovery = http.StatusNotFound, "Evaluation resource not found.", "none"
	case "eval_precondition_required":
		e.Status, e.Message = http.StatusPreconditionRequired, "An evaluation revision is required."
	case "eval_revision_mismatch":
		e.Status, e.Message = http.StatusPreconditionFailed, "The evaluation revision has changed."
	case "eval_idempotency_conflict":
		e.Message, e.Recovery = "The operation key is already bound to another request.", "retry_same_request"
	case "eval_not_ready":
		e.Message, e.Recovery = "The experiment is not ready.", "edit_draft"
	case "eval_preparation_unavailable":
		e.Status, e.Message, e.Recovery = http.StatusServiceUnavailable, "Experiment preparation is temporarily unavailable.", "wait"
	case "eval_pin_mismatch":
		e.Message, e.Recovery = "Required evaluation pins do not match.", "duplicate"
	case "eval_external_control":
		e.Message, e.Recovery = "The experiment has a different execution controller.", "none"
	case "eval_view_changed":
		e.Message = "The selected evaluation view has changed."
	case "eval_member_conflict":
		e.Message, e.Recovery = "The evaluation member has conflicting records.", "inspect_execution"
	case "eval_evidence_unavailable":
		e.Message, e.Recovery = "Required evaluation evidence is unavailable.", "inspect_execution"
	case "eval_producer_stale":
		e.Message, e.Recovery = "The external producer has no recent observation.", "inspect_execution"
	case "eval_budget_exhausted":
		e.Message, e.Recovery = "The frozen experiment allowance is exhausted.", "duplicate"
	case "eval_project_deleting":
		e.Message, e.Recovery = "The evaluation workspace is being deleted.", "wait"
	default:
		return Failure("eval_invalid")
	}
	return e
}
