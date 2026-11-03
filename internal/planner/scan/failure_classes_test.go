package scan

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgconn"
)

type scanFaultArtifacts struct {
	ArtifactStore
	read   func(context.Context, contracts.ArtifactRef, int) (artifacts.Payload, error)
	create func(context.Context, contracts.ArtifactRef, artifacts.Payload) (contracts.ArtifactRef, error)
}

func (s *scanFaultArtifacts) Read(ctx context.Context, runID string, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
	if s.read == nil {
		return s.ArtifactStore.Read(ctx, runID, ref, limit)
	}
	return s.read(ctx, ref, limit)
}

func (s *scanFaultArtifacts) Create(ctx context.Context, runID string, ref contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
	if s.create == nil {
		return s.ArtifactStore.Create(ctx, runID, ref, payload)
	}
	return s.create(ctx, ref, payload)
}

func TestAuditResultReadFailureRetriesWithoutRescanning(t *testing.T) {
	for _, cause := range []error{
		&pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure},
		&pgconn.PgError{Code: persistencepostgres.SQLStateLockNotAvailable},
		&pgconn.PgError{Code: persistencepostgres.SQLStateConnectionFailure},
		context.DeadlineExceeded,
		context.Canceled,
	} {
		t.Run(cause.Error(), func(t *testing.T) {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
				read: func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
					if len(h.invoker.calls) > 0 && strings.HasPrefix(ref.Name, "plan.") {
						return artifacts.Payload{}, cause
					}
					return h.artifacts.Read(ctx, h.invocation.RunID, ref, limit)
				}}
			_, err := h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			if err == nil || failure.Code != "audit_scan_result_unavailable" || !failure.Retryable || !errors.Is(err, cause) {
				t.Fatalf("result read failure = %+v, cause preserved=%v", failure, errors.Is(err, cause))
			}
			if len(h.invoker.calls) != 1 || h.sessions.completeCalls != 0 {
				t.Fatalf("failed result build: calls=%d, completions=%d", len(h.invoker.calls), h.sessions.completeCalls)
			}
			history.prior = []planner.ScanAttempt{{StageExecutionID: history.stageID, StageName: "execute", Terminal: true, State: cloneFactoryValue(h.sessions.state)}}
			h.invocation.StageExecutionID = "result-read-retry"
			history.stageID = h.invocation.StageExecutionID
			h.sessions.state = planner.ScanState{}
			h.factory.artifacts = h.artifacts
			result := h.run(t)
			pkg := auditFactoryResult(t, h, result)
			if result.Outcome != stageresult.StageSucceeded || len(h.invoker.calls) != 1 || !pkg.ExecutionCompleted {
				t.Fatalf("retry rescanned or lost coverage: result=%+v calls=%d", result, len(h.invoker.calls))
			}
		})
	}
}

func TestAuditResultPermanentReadRejectionsDoNotRetry(t *testing.T) {
	for _, cause := range []error{
		blob.ErrIntegrity, blob.ErrTooLarge, artifacts.ErrInvalidScope,
		&pgconn.PgError{Code: persistencepostgres.SQLStateUndefinedColumn},
	} {
		t.Run(cause.Error(), func(t *testing.T) {
			h, _ := newAuditFactoryHarness(t, "sqlmap")
			h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
				read: func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
					if len(h.invoker.calls) > 0 && strings.HasPrefix(ref.Name, "plan.") {
						return artifacts.Payload{}, cause
					}
					return h.artifacts.Read(ctx, h.invocation.RunID, ref, limit)
				}}
			_, err := h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			if err == nil || failure.Code != "audit_scan_result_unavailable" || failure.Retryable || len(h.invoker.calls) != 1 {
				t.Fatalf("permanent read failure=%+v calls=%d", failure, len(h.invoker.calls))
			}
		})
	}
}

func TestAuditResultInvalidRetainedContentStaysPermanent(t *testing.T) {
	for _, fault := range []string{"changed-plan", "oversized-report", "assignment"} {
		t.Run(fault, func(t *testing.T) {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			instance := h.instance(t).(*execution)
			if _, err := instance.Run(t.Context()); err != nil {
				t.Fatal(err)
			}
			attempts, err := history.ReadAuditScanHistory(t.Context(), h.invocation.RunID)
			if err != nil {
				t.Fatal(err)
			}
			if fault == "assignment" {
				instance.audit.task.Scanner = "invalid"
			}
			h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
				read: func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
					payload, err := h.artifacts.Read(ctx, h.invocation.RunID, ref, limit)
					if fault == "changed-plan" && strings.HasPrefix(ref.Name, "plan.") {
						payload.Data = append(payload.Data, ' ')
					}
					if fault == "oversized-report" && h.sessions.state.Jobs[0].Report.SameExact(ref) {
						payload.Data = make([]byte, planner.MaxScanReportBytes+1)
					}
					return payload, err
				}}
			_, err = instance.finishAudit(t.Context(), planner.ScanSessionIdentity{}, attempts)
			failure := planner.FailureFrom(err)
			if err == nil || failure.Code != "audit_scan_result_invalid" || failure.Retryable || len(h.invoker.calls) != 1 {
				t.Fatalf("invalid result failure=%+v calls=%d", failure, len(h.invoker.calls))
			}
		})
	}
}

func TestScanStorageCodeRetryScopeAndCause(t *testing.T) {
	preDispatch := []string{"scan_plan_write_failed", "scan_input_write_failed", "scan_plan_record_failed"}
	auditOnly := []string{
		"audit_scan_source_unavailable", "audit_scan_settings_unavailable",
		"audit_scan_operation_unavailable", "audit_scan_preparation_write_failed", "scan_history_unavailable",
		"audit_scan_result_unavailable", "audit_scan_result_write_failed", "scan_completion_write_failed",
		"scan_session_unavailable", "scan_source_unavailable", "scan_job_claim_unavailable", "scan_job_result_unavailable",
		"scan_saved_plan_unavailable", "scan_saved_input_unavailable",
	}
	causes := []struct {
		name      string
		err       error
		retryable bool
	}{
		{"transient", &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}, true},
		{"ambiguous", &pgconn.PgError{Code: persistencepostgres.SQLStateConnectionFailure}, true},
		{"timeout", context.DeadlineExceeded, true}, {"cancelled", context.Canceled, true},
		{"unavailable", errors.New("blob storage unavailable"), true},
		{"rejected", runstore.ErrInvalid, false}, {"missing", artifacts.ErrArtifactNotFound, false},
		{"conflict", artifacts.ErrArtifactConflict, false}, {"oversized", blob.ErrTooLarge, false},
		{"in-progress", planner.ErrInvocationInProgress, false}, {"no-cause", nil, false},
		{"permanent-sql", &pgconn.PgError{Code: persistencepostgres.SQLStateInsufficientPrivilege}, false},
	}
	for _, audit := range []bool{false, true} {
		h := newFactoryHarness(t, 1)
		if audit {
			h, _ = newAuditFactoryHarness(t, "sqlmap")
		}
		instance := h.instance(t).(*execution)
		for _, code := range append(append([]string{}, preDispatch...), auditOnly...) {
			for _, cause := range causes {
				want := cause.retryable && (audit || code == "scan_plan_write_failed" || code == "scan_input_write_failed" || code == "scan_plan_record_failed")
				failure := instance.scanError(code, cause.err)
				if failure.Retryable != want || !errors.Is(failure, cause.err) && cause.err != nil {
					t.Errorf("audit=%t %s/%s = %+v; want retryable=%t", audit, code, cause.name, failure.Failure, want)
				}
			}
		}
	}
}

func TestAuditStorageFailuresKeepCauseOnRunPath(t *testing.T) {
	for _, phase := range []struct {
		name, code string
	}{
		{"operation", "audit_scan_operation_unavailable"},
		{"openapi", "audit_scan_source_unavailable"},
		{"settings", "audit_scan_settings_unavailable"},
		{"prepared", "audit_scan_preparation_write_failed"},
		{"history", "scan_history_unavailable"},
		{"report", "audit_scan_result_write_failed"},
		{"job-result", "scan_job_result_unavailable"},
		{"job-claim", "scan_job_claim_unavailable"},
	} {
		for _, cause := range []error{
			&pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure},
			&pgconn.PgError{Code: persistencepostgres.SQLStateConnectionFailure},
			blob.ErrIntegrity,
		} {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			switch phase.name {
			case "history":
				history.err = cause
			case "job-result":
				h.sessions.finishErr = cause
			case "job-claim":
				h.sessions.rejectClaim, h.sessions.claimErr = true, cause
			case "prepared", "report":
				h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
					create: func(ctx context.Context, ref contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
						if phase.name == "prepared" && strings.HasPrefix(ref.Name, "prepared.") || phase.name == "report" && len(h.invoker.calls) > 0 {
							return contracts.ArtifactRef{}, cause
						}
						return h.artifacts.Create(ctx, h.invocation.RunID, ref, payload)
					}}
			default:
				h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
					read: func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
						if h.invocation.Context.Artifacts[phase.name].SameExact(ref) {
							return artifacts.Payload{}, cause
						}
						return h.artifacts.Read(ctx, h.invocation.RunID, ref, limit)
					}}
			}
			_, err := h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			wantCalls := 0
			if phase.name == "report" || phase.name == "job-result" {
				wantCalls = 1
			}
			if err == nil || failure.Code != phase.code || failure.Retryable == errors.Is(cause, blob.ErrIntegrity) || !errors.Is(err, cause) || len(h.invoker.calls) != wantCalls {
				t.Fatalf("%s storage failure=%+v, cause=%v, scanner calls=%d", phase.name, failure, err, len(h.invoker.calls))
			}
		}
	}
}

func TestOrdinaryPostDispatchStorageFailureCannotRetry(t *testing.T) {
	for _, phase := range []string{"job-result", "report"} {
		t.Run(phase, func(t *testing.T) {
			h := newFactoryHarness(t, 2)
			cause := &pgconn.PgError{Code: persistencepostgres.SQLStateConnectionFailure}
			wantCode, wantCalls := "scan_job_result_unavailable", 1
			if phase == "job-result" {
				h.sessions.finishErr = cause
			} else {
				wantCode, wantCalls = "scan_report_write_failed", 2
				h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
					create: func(ctx context.Context, ref contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
						if len(h.invoker.calls) > 0 {
							return contracts.ArtifactRef{}, cause
						}
						return h.artifacts.Create(ctx, h.invocation.RunID, ref, payload)
					}}
			}
			_, err := h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			if err == nil || failure.Code != wantCode || failure.Retryable || len(h.invoker.calls) != wantCalls {
				t.Fatalf("post-dispatch %s failure=%+v calls=%d", phase, failure, len(h.invoker.calls))
			}
		})
	}
}

func TestOrdinaryPlanAndInputWriteFailureMayRetryBeforeDispatch(t *testing.T) {
	for _, prefix := range []string{"plan.", "request."} {
		t.Run(prefix, func(t *testing.T) {
			h := newFactoryHarness(t, 1)
			h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
				create: func(ctx context.Context, ref contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
					if strings.HasPrefix(ref.Name, prefix) {
						return contracts.ArtifactRef{}, context.DeadlineExceeded
					}
					return h.artifacts.Create(ctx, h.invocation.RunID, ref, payload)
				}}
			_, err := h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			if err == nil || !failure.Retryable || len(h.invoker.calls) != 0 {
				t.Fatalf("pre-dispatch failure=%+v calls=%d", failure, len(h.invoker.calls))
			}
		})
	}
}

func TestAuditResultBuildHonorsCallerCancellationAndDeadline(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		h, history := newAuditFactoryHarness(t, "sqlmap")
		instance := h.instance(t).(*execution)
		if _, err := instance.Run(t.Context()); err != nil {
			t.Fatal(err)
		}
		attempts, err := history.ReadAuditScanHistory(t.Context(), h.invocation.RunID)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		wantCause := context.Canceled
		if deadline {
			ctx, cancel = context.WithDeadline(t.Context(), time.Time{})
			defer cancel()
			wantCause = context.DeadlineExceeded
		}
		reads := 0
		h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
			read: func(readCtx context.Context, _ contracts.ArtifactRef, _ int) (artifacts.Payload, error) {
				reads++
				if !errors.Is(readCtx.Err(), wantCause) {
					t.Fatalf("result-building context lost caller termination: %v", readCtx.Err())
				}
				return artifacts.Payload{}, readCtx.Err()
			}}
		_, err = instance.finishAudit(ctx, planner.ScanSessionIdentity{}, attempts)
		failure := planner.FailureFrom(err)
		if !errors.Is(err, wantCause) || failure.Code != "audit_scan_result_unavailable" || !failure.Retryable || reads != 0 || h.sessions.completeCalls != 1 {
			t.Fatalf("cancelled result build = %+v, reads=%d, completions=%d", failure, reads, h.sessions.completeCalls)
		}
	}
}

func TestScanResultVerificationRetryScope(t *testing.T) {
	for _, audit := range []bool{false, true} {
		for _, recovered := range []bool{false, true} {
			h := newFactoryHarness(t, 1)
			if audit {
				h, _ = newAuditFactoryHarness(t, "sqlmap")
			}
			if recovered {
				h.run(t)
			}
			inspector, err := newScanInspector(h.invocation.RunID, scanMetadataReader(func(context.Context, contracts.ArtifactRef) (artifacts.Metadata, error) {
				return artifacts.Metadata{}, context.DeadlineExceeded
			}))
			if err != nil {
				t.Fatal(err)
			}
			h.factory.inspector = inspector
			_, err = h.instance(t).Run(t.Context())
			failure := planner.FailureFrom(err)
			if err == nil || failure.Code != "result_artifact_unavailable" || failure.Retryable != audit || len(h.invoker.calls) != 1 {
				t.Fatalf("audit=%t recovered=%t verification=%+v scanner calls=%d", audit, recovered, failure, len(h.invoker.calls))
			}
		}
	}
}

func TestSavedScanReadFailureIsUnavailableNotInvalid(t *testing.T) {
	for _, audit := range []bool{false, true} {
		for _, phase := range []string{"plan", "input"} {
			for _, cause := range []error{context.DeadlineExceeded, artifacts.ErrArtifactNotFound} {
				h := newFactoryHarness(t, 1)
				if audit {
					h, _ = newAuditFactoryHarness(t, "sqlmap")
				}
				h.freeze(t)
				ref := h.sessions.state.Plan.Clone()
				if phase == "input" {
					ref = h.sessions.state.Jobs[0].InputArtifacts["request"]
				}
				h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
					read: func(ctx context.Context, requested contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
						if requested.SameExact(ref) {
							return artifacts.Payload{}, cause
						}
						return h.artifacts.Read(ctx, h.invocation.RunID, requested, limit)
					}}
				_, err := h.instance(t).Run(t.Context())
				failure := planner.FailureFrom(err)
				wantRetry := audit && errors.Is(cause, context.DeadlineExceeded)
				if err == nil || failure.Code != "scan_saved_"+phase+"_unavailable" || failure.Retryable != wantRetry || !errors.Is(err, cause) || len(h.invoker.calls) != 0 {
					t.Fatalf("audit=%t phase=%s read failure=%+v calls=%d", audit, phase, failure, len(h.invoker.calls))
				}
			}
		}
	}
}

func TestSavedScanInputMismatchRemainsInvalid(t *testing.T) {
	for _, audit := range []bool{false, true} {
		h := newFactoryHarness(t, 1)
		if audit {
			h, _ = newAuditFactoryHarness(t, "sqlmap")
		}
		h.freeze(t)
		ref := h.sessions.state.Jobs[0].InputArtifacts["request"]
		payload := h.artifacts.payload(ref)
		payload.Data = append(payload.Data, ' ')
		h.artifacts.values[factoryRefKey(ref)] = payload
		_, err := h.instance(t).Run(t.Context())
		failure := planner.FailureFrom(err)
		if err == nil || failure.Code != "scan_saved_input_invalid" || failure.Retryable || len(h.invoker.calls) != 0 {
			t.Fatalf("audit=%t saved input mismatch=%+v, scanner calls=%d", audit, failure, len(h.invoker.calls))
		}
	}
}
