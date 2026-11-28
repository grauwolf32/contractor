package privateartifacts

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"github.com/jackc/pgx/v5/pgconn"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"strings"
	"testing"
)

func TestPrivateAllocationPreservesNativeFailureClasses(t *testing.T) {
	for _, test := range []struct {
		name      string
		err       error
		status    codes.Code
		code      string
		retryable bool
	}{
		{"package", &auditdomain.ValidationError{Code: auditdomain.CodePackageInvalid, Field: "private-canary"}, codes.InvalidArgument, "audit_package_invalid", false},
		{"capacity", blob.ErrTransferCapacity, codes.ResourceExhausted, "artifact_transfer_capacity", true},
		{"cancelled", fmt.Errorf("private-canary: %w", context.Canceled), codes.Canceled, "cancelled", false},
		{"deadline", fmt.Errorf("private-canary: %w", context.DeadlineExceeded), codes.DeadlineExceeded, "deadline_exceeded", true},
		{"transaction", &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure, Message: "private-canary"}, codes.Aborted, "storage_transaction_conflict", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			intake := &evidenceFailure{err: test.err}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, AuditEvidence: intake}})
			_, err := client.SubmitAuditEvidence(ctx, evidenceRPCRequest(t))
			details := status.Convert(err).Details()
			if status.Code(err) != test.status || len(details) != 1 || details[0].(*pb.ErrorDetail).GetCode() != test.code || details[0].(*pb.ErrorDetail).GetRetryable() != test.retryable || intake.calls != 1 || strings.Contains(err.Error(), "private-canary") {
				t.Fatalf("calls=%d status=%v details=%v", intake.calls, err, details)
			}
		})
	}
}

type evidenceFailure struct {
	err   error
	calls int
}

func (f *evidenceFailure) RecordEvidence(context.Context, controlplane.AllocationGrant, auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, error) {
	f.calls++
	return auditcontract.EvidenceReceipt{}, f.err
}
func evidenceRPCRequest(t *testing.T) *pb.AuditEvidenceRequest {
	t.Helper()
	request := new(pb.AuditEvidenceRequest)
	if err := runtimewire.EncodeDomain(json.RawMessage(`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","itemKey":"item-00","sources":[{"namespace":"inputs","name":"source","revision":"exact"}],"interpretation":"Selected source supports only this claim","testedConditions":[],"limitations":[]}`), request, 8<<20, (8<<20)+rpcEnvelopeBytes); err != nil {
		t.Fatal(err)
	}
	return request
}
func TestPrivateEvidenceClassifiesPermanentDomainRejections(t *testing.T) {
	for _, test := range []struct {
		name   string
		err    error
		status codes.Code
		code   string
	}{
		{"quota", auditstore.ErrEvidenceBudgetExhausted, codes.FailedPrecondition, "audit_evidence_limit_exceeded"},
		{"identity", auditstore.ErrConflict, codes.FailedPrecondition, "audit_evidence_conflict"},
		{"closed", auditstore.ErrPrecondition, codes.FailedPrecondition, "audit_evidence_closed"},
		{"run-gone", runstore.ErrNotFound, codes.FailedPrecondition, "audit_evidence_closed"},
		{"foreign-item", auditstore.ErrNotFound, codes.InvalidArgument, "invalid_request"},
		{"invalid", auditstore.ErrInvalid, codes.InvalidArgument, "invalid_request"},
	} {
		t.Run(test.name, func(t *testing.T) {
			intake := &evidenceFailure{err: test.err}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, AuditEvidence: intake}})
			_, err := client.SubmitAuditEvidence(ctx, evidenceRPCRequest(t))
			details := status.Convert(err).Details()
			if status.Code(err) != test.status || len(details) != 1 || details[0].(*pb.ErrorDetail).GetCode() != test.code || details[0].(*pb.ErrorDetail).GetRetryable() || intake.calls != 1 {
				t.Fatalf("calls=%d status=%v details=%v", intake.calls, err, details)
			}
		})
	}
}
func TestPrivateEvidenceStrictInputNeverReachesTheOwner(t *testing.T) {
	for _, name := range []string{"missing-version", "unknown", "nested-unknown", "missing-content"} {
		t.Run(name, func(t *testing.T) {
			intake := &evidenceFailure{err: errors.New("unexpected owner call")}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, AuditEvidence: intake}})
			request := evidenceRPCRequest(t)
			switch name {
			case "missing-version":
				request.ApiVersion = nil
			case "unknown":
				request.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "nested-unknown":
				request.Sources.Values[0].ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "missing-content":
				request.TestedConditions = nil
			}
			_, err := client.SubmitAuditEvidence(ctx, request)
			if status.Code(err) != codes.InvalidArgument || intake.calls != 0 {
				t.Fatalf("calls=%d error=%v", intake.calls, err)
			}
		})
	}
}
