package controlplane

import (
	"context"
	"errors"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestStartupConfigRequiresMTLSWithoutRegisteringPrincipal(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler := &controlRPCServer{registry: registry, principals: failingPrincipalRegistrar{err: errors.New("bootstrap must not register a principal")}}
	if _, err := handler.Bootstrap(context.Background(), &pb.BootstrapRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unverified bootstrap = %v", err)
	}
	request := &pb.BootstrapRequest{}
	request.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
	if _, err := handler.Bootstrap(registrationPeerContext(), request); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("unknown bootstrap input = %v", err)
	}
	response, err := handler.Bootstrap(registrationPeerContext(), &pb.BootstrapRequest{})
	if err != nil {
		t.Fatal(err)
	}
	var config control.SharedConfig
	if err := runtimewire.Decode(response, &config); err != nil {
		t.Fatal(err)
	}
	expected, err := control.DefaultSharedConfig()
	if err != nil || config != expected {
		t.Fatalf("bootstrap changed canonical policy: %+v / %v", config, err)
	}
	if len(registry.agents) != 0 {
		t.Fatal("bootstrap registered an agent")
	}
}
