package a2a

import (
	"context"
	"crypto/x509"
	"net"
	"testing"
	"time"

	a2agrpc "github.com/a2aproject/a2a-go/v2/a2agrpc/v1"
	a2apb "github.com/a2aproject/a2a-go/v2/a2apb/v1"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/mtlstest"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
)

func nativeSDK(t *testing.T, executor a2asrv.AgentExecutor, options ...grpc.ServerOption) (*Invoker, contracts.WorkerHandle, <-chan string) {
	t.Helper()
	root := t.TempDir()
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	cp, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	tlsConfig, err := mtlstest.AgentServerConfig(mtlstest.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate})
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := x509.ParseCertificate(tlsConfig.Certificates[0].Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(certificate)
	if err != nil {
		t.Fatal(err)
	}
	ids := make(chan string, 32)
	baseOptions := []grpc.ServerOption{grpc.Creds(credentials.NewTLS(tlsConfig)), grpc.UnaryInterceptor(func(ctx context.Context, request any, info *grpc.UnaryServerInfo, next grpc.UnaryHandler) (any, error) {
		md, _ := metadata.FromIncomingContext(ctx)
		values := md.Get("x-request-id")
		if len(values) == 1 {
			select {
			case ids <- values[0]:
			default:
			}
		}
		return next(ctx, request)
	})}
	server := grpc.NewServer(append(baseOptions, options...)...)
	a2apb.RegisterA2AServiceServer(server, a2agrpc.NewHandler(a2asrv.NewHandler(executor)))
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	t.Cleanup(func() { server.Stop(); <-done; _ = listener.Close() })
	invoker, err := NewMTLS(mtls.Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate}, 3*time.Second, Options{PollInterval: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	handle := workerHandle(listener.Addr().String())
	handle.RuntimeAgentID = principal
	return invoker, handle, ids
}
