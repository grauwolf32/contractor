package privateartifacts

import (
	"context"
	"encoding/json"
	"errors"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"testing"
)

type judgmentFailure struct {
	err   error
	calls int
}

func (f *judgmentFailure) RecordJudgments(context.Context, controlplane.AllocationGrant, auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, error) {
	f.calls++
	return auditcontract.JudgmentReceipt{}, f.err
}
func judgmentRPCRequest(t *testing.T) *pb.AuditJudgmentsRequest {
	t.Helper()
	request := new(pb.AuditJudgmentsRequest)
	if err := runtimewire.EncodeDomain(json.RawMessage(`{"apiVersion":"contractor/v1alpha1","invocationId":"invocation","submissionId":"call","mode":"single","items":[{"baseGeneration":0,"conclusion":{"item_key":"item-00","rationale":"Scoped judgment","tested_conditions":[],"evidence":[],"limitations":[]}}]}`), request, 8<<20, (8<<20)+rpcEnvelopeBytes); err != nil {
		t.Fatal(err)
	}
	return request
}
func TestPrivateJudgmentClassifiesPermanentDomainRejections(t *testing.T) {
	for _, test := range []struct {
		name   string
		err    error
		status codes.Code
		code   string
	}{
		{"quota", auditstore.ErrEvidenceBudgetExhausted, codes.FailedPrecondition, "audit_result_limit_exceeded"},
		{"identity", auditstore.ErrConflict, codes.FailedPrecondition, "audit_result_conflict"},
		{"closed", auditstore.ErrPrecondition, codes.FailedPrecondition, "audit_result_closed"},
		{"run-gone", runstore.ErrNotFound, codes.FailedPrecondition, "audit_result_closed"},
		{"foreign-item", auditstore.ErrNotFound, codes.InvalidArgument, "invalid_request"},
		{"invalid", auditstore.ErrInvalid, codes.InvalidArgument, "invalid_request"},
	} {
		t.Run(test.name, func(t *testing.T) {
			intake := &judgmentFailure{err: test.err}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, AuditJudgments: intake}})
			_, err := client.SubmitAuditJudgments(ctx, judgmentRPCRequest(t))
			details := status.Convert(err).Details()
			if status.Code(err) != test.status || len(details) != 1 || details[0].(*pb.ErrorDetail).GetCode() != test.code || details[0].(*pb.ErrorDetail).GetRetryable() || intake.calls != 1 {
				t.Fatalf("calls=%d status=%v details=%v", intake.calls, err, details)
			}
		})
	}
}
func TestPrivateJudgmentStrictInputNeverReachesTheOwner(t *testing.T) {
	for _, name := range []string{"missing-version", "unknown", "nested-unknown", "missing-content"} {
		t.Run(name, func(t *testing.T) {
			intake := &judgmentFailure{err: errors.New("unexpected owner call")}
			client, ctx := allocationRPCFixture(t, &handler{dependencies: Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, AuditJudgments: intake}})
			request := judgmentRPCRequest(t)
			switch name {
			case "missing-version":
				request.ApiVersion = nil
			case "unknown":
				request.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "nested-unknown":
				request.Items.Values[0].Conclusion.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
			case "missing-content":
				request.Items.Values[0].BaseGeneration = nil
			}
			_, err := client.SubmitAuditJudgments(ctx, request)
			if status.Code(err) != codes.InvalidArgument || intake.calls != 0 {
				t.Fatalf("calls=%d error=%v", intake.calls, err)
			}
		})
	}
}
