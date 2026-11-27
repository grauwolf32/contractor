package controlplane

import (
	"context"
	"errors"
	"net/http"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/mtls"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
)

type PrincipalRegistrar interface {
	Register(context.Context, string) (agentprincipal.Principal, error)
}

// NewGRPCHandler serves typed control operations on the private HTTP/2 listener.
// The listener owns connections and shutdown while unmigrated operation families
// still share that listener. There is no fallback from an RPC to an HTTP route.
func NewGRPCHandler(registry Registry, principals PrincipalRegistrar) (http.Handler, error) {
	if registry == nil || principals == nil {
		return nil, errors.New("Control Plane gRPC dependencies are required")
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(runtimewire.MaxControlBytes),
		grpc.MaxSendMsgSize(runtimewire.MaxControlBytes),
		grpc.MaxHeaderListSize(runtimewire.MaxMetadataBytes),
	)
	pb.RegisterControlPlaneServer(server, &controlRPCServer{registry: registry, principals: principals})
	return &boundedControlRPC{next: server, permits: make(chan struct{}, runtimewire.MaxConcurrentRPCs)}, nil
}

type controlRPCServer struct {
	pb.UnimplementedControlPlaneServer
	registry   Registry
	principals PrincipalRegistrar
}

func (h *controlRPCServer) Register(ctx context.Context, request *pb.AgentRegistration) (*pb.AgentRegistrationResponse, error) {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return nil, runtimewire.Error(ctx, codes.Unauthenticated, "mtls_required", false)
	}
	tls, ok := p.AuthInfo.(credentials.TLSInfo)
	if !ok {
		return nil, runtimewire.Error(ctx, codes.Unauthenticated, "mtls_required", false)
	}
	identity, err := mtls.RuntimeAgentIDFromConnection(&tls.State)
	if err != nil {
		return nil, runtimewire.Error(ctx, codes.Unauthenticated, "mtls_required", false)
	}
	correlation := requestid.Ensure(ctx)
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		values := md.Get("x-request-id")
		if len(values) == 1 && requestid.Valid(values[0]) {
			correlation = values[0]
		}
	}
	ctx = requestid.With(ctx, correlation)
	_ = grpc.SetHeader(ctx, metadata.Pairs("x-request-id", correlation))
	registration, err := runtimewire.Registration(request)
	if err != nil {
		return nil, runtimewire.Error(ctx, codes.InvalidArgument, "invalid_request", false)
	}
	principal, err := h.principals.Register(ctx, identity)
	if err != nil {
		return nil, controlRPCError(ctx, err)
	}
	_, err = h.registry.RegisterAuthenticated(AuthenticatedPrincipal{
		RuntimeAgentID: principal.RuntimeAgentID,
		Labels:         principal.Labels, LabelRevision: principal.LabelRevision,
	}, registration)
	if err != nil {
		return nil, controlRPCError(ctx, err)
	}
	response, err := runtimewire.RegistrationResponse(h.registry.RegistrationResponse())
	if err != nil {
		return nil, runtimewire.Error(ctx, codes.Internal, "internal_error", false)
	}
	return response, nil
}

func controlRPCError(ctx context.Context, err error) error {
	switch {
	case errors.Is(err, context.Canceled):
		return runtimewire.Error(ctx, codes.Canceled, "cancelled", false)
	case errors.Is(err, context.DeadlineExceeded):
		return runtimewire.Error(ctx, codes.DeadlineExceeded, "deadline_exceeded", true)
	case persistencepostgres.IsTransactionConflict(err):
		return runtimewire.Error(ctx, codes.Aborted, "storage_transaction_conflict", true)
	case errors.Is(err, ErrInvalidRequest), errors.Is(err, contracts.ErrValidation), errors.Is(err, runtimeconfig.ErrInvalid):
		return runtimewire.Error(ctx, codes.InvalidArgument, "invalid_request", false)
	case errors.Is(err, ErrRegistrationConflict), errors.Is(err, ErrHeartbeatOutOfOrder):
		return runtimewire.Error(ctx, codes.FailedPrecondition, "conflict", true)
	default:
		return runtimewire.Error(ctx, codes.Unavailable, "internal_error", true)
	}
}

// Admission precedes grpc-go message decoding, bounding aggregate request memory.
type boundedControlRPC struct {
	next    http.Handler
	permits chan struct{}
}

func (h *boundedControlRPC) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	select {
	case h.permits <- struct{}{}:
		defer func() { <-h.permits }()
		h.next.ServeHTTP(w, r)
	default:
		w.Header().Set("Content-Type", "application/grpc")
		w.Header().Set("Grpc-Status", "8")
		w.Header().Set("Grpc-Message", "control_rpc_capacity")
		w.WriteHeader(http.StatusOK)
	}
}
