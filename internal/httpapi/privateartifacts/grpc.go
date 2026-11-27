package privateartifacts

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
)

const allocationIDHeader = "X-Contractor-Allocation-ID"
const rpcEnvelopeBytes = 1024

type allocationRPCContext struct {
	grant           controlplane.AllocationGrant
	identity        authenticatedRuntime
	controller      *http.ResponseController
	releaseTransfer func()
}
type allocationRPCContextKey struct{}

type allocationRPCServer struct {
	pb.UnimplementedAllocationServer
	owner *handler
}

// NewGRPCHandler exposes named allocation operations through their existing
// domain owners. Authority and aggregate admission precede message decoding.
func NewGRPCHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.Transfers == nil || dependencies.Registry == nil || dependencies.Artifacts == nil {
		return nil, errors.New("private allocation RPC dependencies are incomplete")
	}
	owner := &handler{dependencies: dependencies}
	if dependencies.GatewayRecovery != nil {
		owner.recovery = dependencies.GatewayRecovery
	}
	return newAllocationGRPCHandler(owner)
}

func newAllocationGRPCHandler(owner *handler) (http.Handler, error) {
	dependencies := owner.dependencies
	rpc := &allocationRPCServer{owner: owner}
	servers := make(map[int]*grpc.Server)
	for _, limit := range []int{maximumRecoveryRequestBytes, contracts.MaxPlannerOperationBytes, auditcontract.MaximumDocumentBytes, projectsitemap.MaxBatchBytes, 1 << 20} {
		server := grpc.NewServer(
			grpc.MaxRecvMsgSize(limit+rpcEnvelopeBytes),
			grpc.MaxSendMsgSize(limit+rpcEnvelopeBytes),
			grpc.MaxHeaderListSize(runtimewire.MaxMetadataBytes),
			grpc.UnaryInterceptor(allocationRPCDecoded),
		)
		pb.RegisterAllocationServer(server, rpc)
		servers[limit] = server
	}
	permits := make(chan struct{}, runtimewire.MaxConcurrentRPCs)
	native := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		correlation := requestid.Ensure(r.Context())
		if dependencies.NewRequestID != nil {
			if generated, err := dependencies.NewRequestID(); err == nil && requestid.Valid(generated) {
				correlation = generated
			}
		}
		r = r.WithContext(requestid.With(r.Context(), correlation))
		w.Header().Set(requestid.Header, correlation)
		limit := allocationRPCLimit(r.URL.Path)
		if limit == 0 {
			runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.Unimplemented, "operation_not_found", false))
			return
		}
		identity, err := mtls.RuntimeAgentIDFromConnection(r.TLS)
		if err != nil {
			runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.Unauthenticated, "mtls_required", false))
			return
		}
		if values := r.Header.Values(requestid.Header); len(values) == 1 && requestid.Valid(values[0]) {
			r = r.WithContext(requestid.With(r.Context(), values[0]))
			w.Header().Set(requestid.Header, values[0])
		}
		instances, allocations := r.Header.Values(RuntimeInstanceHeader), r.Header.Values(allocationIDHeader)
		if len(instances) != 1 || len(instances[0]) > 256 || strings.TrimSpace(instances[0]) == "" || instances[0] != strings.TrimSpace(instances[0]) {
			runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.InvalidArgument, "invalid_runtime_instance", false))
			return
		}
		if len(allocations) != 1 || !contracts.ValidResourceID(allocations[0]) {
			runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.InvalidArgument, "invalid_request", false))
			return
		}
		grant, err := dependencies.Registry.GetGrant(allocations[0])
		if err != nil || grant.AllocationID != allocations[0] || grant.RuntimeAgentID != identity || grant.RuntimeInstanceID != instances[0] {
			failure := allocationDomainError(r.Context(), controlplane.ErrAllocationNotFound)
			if r.URL.Path == pb.Allocation_PlannerOperation_FullMethodName || r.URL.Path == pb.Allocation_GatewayRecovery_FullMethodName {
				failure = runtimewire.Error(r.Context(), codes.FailedPrecondition, "allocation_unavailable", false)
			}
			runtimewire.WriteError(w, failure)
			return
		}
		var releaseTransfer func()
		if isAuditRPC(r.URL.Path) {
			admitted, release, admissionErr := dependencies.Transfers.AcquireTransfer(r.Context())
			if admissionErr != nil {
				runtimewire.WriteError(w, allocationDomainError(r.Context(), admissionErr))
				return
			}
			var once sync.Once
			releaseTransfer = func() { once.Do(release) }
			defer releaseTransfer()
			r = r.WithContext(admitted)
		}
		select {
		case permits <- struct{}{}:
			defer func() { <-permits }()
		default:
			runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.ResourceExhausted, "allocation_rpc_capacity", true))
			return
		}
		controller := http.NewResponseController(w)
		deadline := time.Now().Add(artifacttransfer.Duration(int64(limit + rpcEnvelopeBytes)))
		_ = controller.SetReadDeadline(deadline)
		_ = controller.SetWriteDeadline(deadline)
		scope := allocationRPCContext{grant: grant, identity: authenticatedRuntime{principalID: identity, instanceID: instances[0]}, controller: controller, releaseTransfer: releaseTransfer}
		ctx := context.WithValue(r.Context(), allocationRPCContextKey{}, scope)
		servers[limit].ServeHTTP(w, r.WithContext(ctx))
	})
	return native, nil
}

func allocationRPCLimit(method string) int {
	switch method {
	case pb.Allocation_PlannerOperation_FullMethodName:
		return contracts.MaxPlannerOperationBytes
	case pb.Allocation_GatewayRecovery_FullMethodName:
		return maximumRecoveryRequestBytes
	case pb.Allocation_SubmitAuditEvidence_FullMethodName, pb.Allocation_SubmitAuditJudgments_FullMethodName:
		return auditcontract.MaximumDocumentBytes
	case pb.Allocation_MutateSitemap_FullMethodName, pb.Allocation_SnapshotSitemap_FullMethodName, pb.Allocation_CollectSitemap_FullMethodName:
		return projectsitemap.MaxBatchBytes
	case pb.Allocation_ListArtifacts_FullMethodName, pb.Allocation_ReadArtifactMetadata_FullMethodName, pb.Allocation_CheckArtifactAccess_FullMethodName, pb.Allocation_ListFindings_FullMethodName, pb.Allocation_ReadFinding_FullMethodName, pb.Allocation_SelectFindings_FullMethodName:
		return 1 << 20
	default:
		return 0
	}
}

func allocationRPCDecoded(ctx context.Context, request any, _ *grpc.UnaryServerInfo, next grpc.UnaryHandler) (any, error) {
	scope, ok := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	if !ok {
		return nil, runtimewire.Error(ctx, codes.Unauthenticated, "mtls_required", false)
	}
	// Reading and storage have separate budgets. A completed body must not
	// leave an expired socket deadline cancelling the domain operation.
	_ = scope.controller.SetReadDeadline(time.Time{})
	_ = scope.controller.SetWriteDeadline(time.Time{})
	response, err := next(ctx, request)
	if scope.releaseTransfer != nil {
		scope.releaseTransfer()
	}
	var size int64
	if message, ok := response.(proto.Message); ok {
		size = int64(proto.Size(message))
	}
	_ = scope.controller.SetWriteDeadline(time.Now().Add(artifacttransfer.Duration(size)))
	return response, err
}

func (h *allocationRPCServer) PlannerOperation(ctx context.Context, request *pb.PlannerOperationRequest) (*pb.PlannerOperationResponse, error) {
	scope := ctx.Value(allocationRPCContextKey{}).(allocationRPCContext)
	grant := scope.grant
	if grant.Lost || grant.WriteFenced {
		return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "allocation_unavailable", false)
	}
	var input contracts.PlannerOperation
	if runtimewire.DecodeDomain(request, &input, contracts.MaxPlannerOperationBytes, contracts.MaxPlannerOperationBytes+rpcEnvelopeBytes) != nil || input.Validate() != nil {
		return nil, runtimewire.Error(ctx, codes.InvalidArgument, "invalid_request", false)
	}
	if h.owner.dependencies.PlannerOperations == nil {
		return nil, runtimewire.Error(ctx, codes.Unavailable, "planner_execution_failed", true)
	}
	err := h.owner.dependencies.Registry.WithWriteGrant(grant.AllocationID, func(current controlplane.AllocationGrant) error {
		if current.Lost || current.WriteFenced || current.StageExecutionID != grant.StageExecutionID || current.RuntimeInstanceID != grant.RuntimeInstanceID || current.RuntimeAgentID != scope.identity.principalID {
			return controlplane.ErrAllocationLost
		}
		return nil
	})
	if err != nil {
		return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "allocation_unavailable", false)
	}
	result, err := h.owner.dependencies.PlannerOperations.ExecuteOperation(ctx, grant.StageExecutionID, grant.AllocationID, input)
	if err != nil {
		return nil, runtimewire.Error(ctx, codes.FailedPrecondition, "planner_operation_invalid", false)
	}
	response := new(pb.PlannerOperationResponse)
	if result.Validate() != nil || runtimewire.EncodeDomain(result, response, contracts.MaxPlannerOperationBytes, contracts.MaxPlannerOperationBytes+rpcEnvelopeBytes) != nil {
		return nil, runtimewire.Error(ctx, codes.Unavailable, "planner_execution_failed", true)
	}
	return response, nil
}
