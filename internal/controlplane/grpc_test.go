package controlplane

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/mtls"
	statuspb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func registrationRPCFixture() *pb.AgentRegistration {
	return &pb.AgentRegistration{
		ApiVersion: proto.String(contracts.APIVersion), InstanceId: proto.String("agent-grpc"),
		SoftwareVersion: proto.String("0.1.0"), StartedAt: proto.String("2026-08-29T12:00:00Z"),
		ControlUrl: proto.String("https://agent-grpc.example:9443"), A2AUrl: proto.String("https://agent-grpc.example:9444"),
		SupportedRuntimes:        &pb.ListOfString{Values: []string{"adk@1"}},
		SupportedToolsets:        &pb.ListOfToolsetCapability{Values: []*pb.ToolsetCapability{{Ref: proto.String("run-artifacts@1"), Tools: &pb.ListOfString{Values: []string{"list_artifacts", "read_artifact", "write_artifact"}}}}},
		SupportedSandboxProfiles: &pb.ListOfString{Values: []string{"local-workdir@1"}},
		SupportedRuntimeAdapters: &pb.ListOfString{}, ObservedState: proto.String("idle"),
		AllocationSlots: proto.Int64(1), Allocations: &pb.ListOfObservedAllocation{},
	}
}

func registrationPeerContext() context.Context {
	cert := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("test-runtime-key")}
	return peer.NewContext(context.Background(), &peer.Peer{AuthInfo: credentials.TLSInfo{State: tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{cert}, VerifiedChains: [][]*x509.Certificate{{cert}},
	}}})
}

func TestPrivateRegistrationRequiresVerifiedMTLSAndStrictProtobuf(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler := &controlRPCServer{registry: registry, principals: testPrincipalRegistrar{}}
	untrusted := metadata.NewIncomingContext(context.Background(), metadata.Pairs("x-request-id", "attacker-chosen"))
	_, err := handler.Register(untrusted, registrationRPCFixture())
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unauthenticated status = %v", err)
	}
	details := status.Convert(err).Details()
	if len(details) != 1 || details[0].(*pb.ErrorDetail).GetRequestId() == "attacker-chosen" {
		t.Fatalf("untrusted correlation = %v", details)
	}
	for _, tc := range []struct {
		name   string
		change func(*pb.AgentRegistration)
	}{
		{"missing collection", func(v *pb.AgentRegistration) { v.SupportedRuntimeAdapters = nil }},
		{"unknown top level", func(v *pb.AgentRegistration) { v.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01}) }},
		{"unknown nested", func(v *pb.AgentRegistration) {
			v.SupportedToolsets.Values[0].ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
		}},
		{"explicit invalid zero", func(v *pb.AgentRegistration) { v.AllocationSlots = proto.Int64(0) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := registrationRPCFixture()
			tc.change(request)
			_, err := handler.Register(registrationPeerContext(), request)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("status = %v", err)
			}
			if _, err := registry.GetAgent("agent-grpc"); !errors.Is(err, ErrAgentNotFound) {
				t.Fatalf("invalid request mutated registry: %v", err)
			}
		})
	}
	response, err := handler.Register(registrationPeerContext(), registrationRPCFixture())
	if err != nil || response.GetHeartbeatIntervalSeconds() != 10 || response.GetConfirmedLeaseSeconds() != 60 || response.GetApiVersion() != contracts.APIVersion {
		t.Fatalf("response = %v, %v", response, err)
	}
	expected, _ := mtls.RuntimeAgentID(&x509.Certificate{RawSubjectPublicKeyInfo: []byte("test-runtime-key")})
	registry.mu.Lock()
	principal := clonePrincipal(registry.agents["agent-grpc"].principal)
	registry.mu.Unlock()
	if principal.RuntimeAgentID != expected || principal.LabelRevision != 1 || principal.Labels == nil {
		t.Fatalf("lost principal authority: %+v", principal)
	}
}

func TestPrivateRegistryGRPCMTLSRejectsForeignCA(t *testing.T) {
	trusted, foreign := generatePKI(t, "trusted"), generatePKI(t, "foreign")
	registry := newTestRegistry(t, newTestClock())
	handler, err := NewGRPCHandler(registry, testPrincipalRegistrar{})
	if err != nil {
		t.Fatal(err)
	}
	config, err := mtls.ControlPlaneServerConfig(trusted.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(handler)
	server.EnableHTTP2 = true
	server.TLS = config
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.StartTLS()
	defer server.Close()
	for _, tc := range []struct {
		name    string
		files   mtls.Files
		allowed bool
	}{{"trusted", trusted.agent, true}, {"foreign", mtls.Files{Certificate: foreign.agent.Certificate, PrivateKey: foreign.agent.PrivateKey, CA: trusted.agent.CA}, false}} {
		t.Run(tc.name, func(t *testing.T) {
			clientTLS, err := mtls.ControlPlaneEndpointClientConfig(tc.files)
			if err != nil {
				t.Fatal(err)
			}
			conn, err := grpc.NewClient(strings.TrimPrefix(server.URL, "https://"), grpc.WithTransportCredentials(credentials.NewTLS(clientTLS)), grpc.WithDisableRetry())
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			ctx = metadata.AppendToOutgoingContext(ctx, "x-request-id", "request_grpc_test")
			request := registrationRPCFixture()
			request.InstanceId = proto.String(tc.name)
			var headers metadata.MD
			response, err := pb.NewControlPlaneClient(conn).Register(ctx, request, grpc.Header(&headers))
			if tc.allowed {
				if err != nil || response.GetApiVersion() != contracts.APIVersion || len(headers.Get("x-request-id")) != 1 || headers.Get("x-request-id")[0] != "request_grpc_test" {
					t.Fatalf("response=%v headers=%v error=%v", response, headers, err)
				}
			} else {
				if err == nil {
					t.Fatal("foreign CA reached handler")
				}
				if _, err := registry.GetAgent(tc.name); !errors.Is(err, ErrAgentNotFound) {
					t.Fatalf("foreign principal mutated registry: %v", err)
				}
			}
		})
	}
}

func TestPrivateRegistrationRejectionsCarryTheRetryVerdict(t *testing.T) {
	for _, tc := range []struct {
		name      string
		err       error
		status    codes.Code
		code      string
		retryable bool
	}{
		{"validation", fmt.Errorf("%w: invalid", ErrInvalidRequest), codes.InvalidArgument, "invalid_request", false},
		{"conflict", ErrRegistrationConflict, codes.FailedPrecondition, "conflict", true},
		{"internal", errors.New("secret database error"), codes.Unavailable, "internal_error", true},
		{"cancelled", context.Canceled, codes.Canceled, "cancelled", false},
		{"deadline", context.DeadlineExceeded, codes.DeadlineExceeded, "deadline_exceeded", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler := &controlRPCServer{registry: newTestRegistry(t, newTestClock()), principals: failingPrincipalRegistrar{err: tc.err}}
			_, err := handler.Register(registrationPeerContext(), registrationRPCFixture())
			st := status.Convert(err)
			details := st.Details()
			if st.Code() != tc.status || len(details) != 1 {
				t.Fatalf("status = %v", st)
			}
			detail, ok := details[0].(*pb.ErrorDetail)
			if !ok || detail.GetCode() != tc.code || detail.GetRetryable() != tc.retryable || detail.GetRequestId() == "" || strings.Contains(st.Message(), "secret") {
				t.Fatalf("unsafe/invalid detail: %v", st)
			}
		})
	}
}

func TestPrivateGRPCAdmissionRejectsBeforeReadingRequest(t *testing.T) {
	permits := make(chan struct{}, 32)
	for range cap(permits) {
		permits <- struct{}{}
	}
	h := &boundedControlRPC{next: http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("saturated handler dispatched") }), permits: permits}
	response := httptest.NewRecorder()
	h.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/contractor.runtime.v1alpha1.ControlPlane/Register", nil))
	if response.Code != http.StatusOK || response.Header().Get("Grpc-Status") != "8" {
		t.Fatalf("admission response: %d %s", response.Code, response.Header().Get("Grpc-Status"))
	}
	encoded, err := base64.RawStdEncoding.DecodeString(response.Header().Get("Grpc-Status-Details-Bin"))
	if err != nil {
		t.Fatal(err)
	}
	var wireStatus statuspb.Status
	if err := proto.Unmarshal(encoded, &wireStatus); err != nil {
		t.Fatal(err)
	}
	details := status.FromProto(&wireStatus).Details()
	if wireStatus.Code != int32(codes.ResourceExhausted) || len(details) != 1 {
		t.Fatalf("admission details: %v", &wireStatus)
	}
	detail, ok := details[0].(*pb.ErrorDetail)
	if !ok || detail.GetCode() != "control_rpc_capacity" || !detail.GetRetryable() || detail.GetRequestId() == "" {
		t.Fatalf("admission retry verdict: %v", details)
	}
	<-permits
	called := false
	h.next = http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true })
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/", nil))
	if !called || len(permits) != 31 {
		t.Fatal("permit was not released")
	}
}

func TestHeartbeatRPCPreservesPrincipalAndSequenceAuthority(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler := &controlRPCServer{registry: registry, principals: testPrincipalRegistrar{}}
	ctx := registrationPeerContext()
	if _, err := handler.Register(ctx, registrationRPCFixture()); err != nil {
		t.Fatal(err)
	}
	request := &pb.AgentHeartbeat{ApiVersion: proto.String(contracts.APIVersion), InstanceId: proto.String("agent-grpc"), HeartbeatSeq: proto.Uint64(1), EchoedAckSeq: proto.Uint64(0), ObservedState: proto.String("idle"), Allocations: &pb.ListOfObservedAllocation{}}
	if _, err := handler.Heartbeat(context.Background(), request); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unverified heartbeat: %v", err)
	}
	for _, mutate := range []func(*pb.AgentHeartbeat){
		func(v *pb.AgentHeartbeat) { v.HeartbeatSeq = nil },
		func(v *pb.AgentHeartbeat) { v.EchoedAckSeq = nil },
		func(v *pb.AgentHeartbeat) { v.Allocations = nil },
		func(v *pb.AgentHeartbeat) { v.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 1}) },
	} {
		invalid := proto.Clone(request).(*pb.AgentHeartbeat)
		mutate(invalid)
		if _, err := handler.Heartbeat(ctx, invalid); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("malformed heartbeat: %v", err)
		}
	}
	response, err := handler.Heartbeat(ctx, request)
	if err != nil || response.GetAckSeq() != 1 || response.GetAction() != "continue" {
		t.Fatalf("heartbeat: %v, %v", response, err)
	}
	// Exact re-delivery uses the original sequence, without a new lease authority.
	replay, err := handler.Heartbeat(ctx, request)
	if err != nil || !proto.Equal(response, replay) {
		t.Fatalf("heartbeat replay: %v, %v", replay, err)
	}
	foreign := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("foreign-runtime-key")}
	foreignCtx := peer.NewContext(context.Background(), &peer.Peer{AuthInfo: credentials.TLSInfo{State: tls.ConnectionState{PeerCertificates: []*x509.Certificate{foreign}, VerifiedChains: [][]*x509.Certificate{{foreign}}}}})
	response, err = handler.Heartbeat(foreignCtx, request)
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("foreign heartbeat: %v, %v", response, err)
	}
	request.InstanceId = proto.String("stale-instance")
	response, err = handler.Heartbeat(ctx, request)
	if err != nil || response.GetAction() != "reregister" {
		t.Fatalf("stale heartbeat: %v, %v", response, err)
	}
}
