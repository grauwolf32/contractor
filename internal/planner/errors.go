package planner

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

var ErrInvocationInProgress = errors.New("Planner invocation is already in progress")

// Failure is safe to persist and return to Workflow Scheduler. Transport and
// provider error strings are retained only as an unexported wrapped cause.
type Failure struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

type Error struct {
	Failure
	cause error
}

func NewError(code, message string, retryable bool, cause error) *Error {
	return &Error{
		Failure: Failure{Code: code, Message: message, Retryable: retryable},
		cause:   cause,
	}
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error { return e.cause }

func FailureFrom(err error) Failure {
	var plannerError *Error
	if errors.As(err, &plannerError) {
		return plannerError.Failure
	}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return Failure{
			Code: "worker_deadline_exceeded", Message: "Worker invocation deadline expired",
			Retryable: true,
		}
	case errors.Is(err, context.Canceled):
		return Failure{
			Code: "planner_cancelled", Message: "Planner invocation was cancelled",
			Retryable: true,
		}
	default:
		return Failure{
			Code: "worker_unavailable", Message: "Worker invocation failed", Retryable: true,
		}
	}
}

func validateFailure(value Failure) error {
	if strings.TrimSpace(value.Code) == "" || strings.TrimSpace(value.Message) == "" {
		return errors.New("Planner failure code and message are required")
	}
	return nil
}
