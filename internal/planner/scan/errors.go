package scan

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
)

func (p *execution) scanError(code string, cause error) *planner.Error {
	retryable := false
	switch code {
	case "scan_plan_write_failed", "scan_input_write_failed", "scan_plan_record_failed":
		// No job can be claimed before all inputs and InitializeScan succeed.
		retryable = retryableScanCause(cause)
	case "audit_scan_source_unavailable", "audit_scan_settings_unavailable",
		"audit_scan_operation_unavailable",
		"audit_scan_preparation_write_failed", "audit_scan_result_unavailable",
		"audit_scan_result_write_failed", "scan_history_unavailable",
		"scan_saved_plan_unavailable", "scan_saved_input_unavailable",
		"scan_session_unavailable", "scan_source_unavailable", "scan_job_claim_unavailable",
		"scan_job_result_unavailable", "scan_completion_write_failed":
		// Audit history reuses a completed/unknown attempt across Stage identities.
		// Ordinary sessions cannot make that guarantee, even during recovery reads.
		retryable = p.invocation.Stage.ScanPreparation != nil && retryableScanCause(cause)
	}
	return planner.NewError(code, "Scan planning or execution could not be completed", retryable, cause)
}

// A retained input read leaves its content unverified; semantic mismatches in
// validateSavedInputs are separate permanent rejections.
type savedInputReadError struct{ cause error }

func (e *savedInputReadError) Error() string { return "saved scan input could not be read" }
func (e *savedInputReadError) Unwrap() error { return e.cause }

func retryableScanCause(cause error) bool {
	if cause == nil {
		return false
	}
	// Classify centrally, preserving context precedence over a database condition.
	switch persistencepostgres.Classify(cause) {
	case persistencepostgres.ClassCancelled, persistencepostgres.ClassTimeout,
		persistencepostgres.ClassTransactionConflict, persistencepostgres.ClassUnavailable:
		return true
	case persistencepostgres.ClassNone:
	default:
		return false
	}
	if artifacts.IsExactReferenceRejection(cause) || errors.Is(cause, artifacts.ErrArtifactConflict) || errors.Is(cause, artifacts.ErrRunRevisionLimit) || errors.Is(cause, planner.ErrInvocationInProgress) {
		return false
	}
	var failure *planner.Error
	if errors.As(cause, &failure) {
		return failure.Retryable
	}
	if classified, ok := apierror.Of(cause); ok {
		return classified.ErrorKind().Retryable()
	}
	// Unclassified Artifact/blob I/O failures leave exact content unverified.
	return true
}

// resultError retains shared result-verification codes but narrows their retry
// scope for ordinary scans: a fresh Stage would dispatch its jobs again.
func (p *execution) resultError(err error) error {
	if err == nil || p.invocation.Stage.ScanPreparation != nil {
		return err
	}
	failure := planner.FailureFrom(err)
	return planner.NewError(failure.Code, failure.Message, false, err)
}
