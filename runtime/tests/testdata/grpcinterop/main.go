// V490-001 probe: use the official Go A2A gRPC adapter against the real
// Python SDK handler. Test-only executable, outside the production package graph.
package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	sdk "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	a2agrpc "github.com/a2aproject/a2a-go/v2/a2agrpc/v1"
	"github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/mtls"
	"golang.org/x/net/http2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

type config struct{ Address, Certificate, Key, CA, Principal, Mode string }

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	var cfg config
	if err := json.NewDecoder(os.Stdin).Decode(&cfg); err != nil {
		return err
	}
	if strings.HasPrefix(cfg.Mode, "control") {
		return serveControl(cfg)
	}
	if cfg.Mode == "serve" {
		return serve(cfg)
	}
	tlsConfig, err := mtls.ControlPlaneEndpointClientConfig(mtls.Files{Certificate: cfg.Certificate, PrivateKey: cfg.Key, CA: cfg.CA})
	if err != nil {
		return err
	}
	tlsConfig, err = mtls.BindRuntimeAgentPrincipal(tlsConfig, cfg.Principal)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client, err := a2aclient.NewFromEndpoints(ctx, []*sdk.AgentInterface{{URL: cfg.Address, ProtocolBinding: sdk.TransportProtocolGRPC, ProtocolVersion: "1.0", Tenant: "allocation-probe"}}, a2agrpc.WithGRPCTransport(grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)), grpc.WithDisableRetry()))
	if err != nil {
		return err
	}
	defer client.Destroy()
	send := func(objective string) (*sdk.Task, error) {
		part := sdk.NewDataPart(map[string]any{"apiVersion": "contractor/v1alpha1", "subtaskId": "0", "objective": objective, "instructions": "probe", "parameters": map[string]any{}, "artifacts": map[string]any{}, "deadline": time.Now().Add(time.Minute).UTC().Format(time.RFC3339Nano)})
		part.MediaType = "application/vnd.contractor.stage-content+json"
		result, err := client.SendMessage(ctx, &sdk.SendMessageRequest{Tenant: "allocation-probe", Message: sdk.NewMessage(sdk.MessageRoleUser, part), Config: &sdk.SendMessageConfig{ReturnImmediately: true}})
		if err != nil {
			return nil, err
		}
		task, ok := result.(*sdk.Task)
		if !ok {
			return nil, errors.New("SendMessage did not return a Task")
		}
		return task, nil
	}
	task, err := send("complete")
	if cfg.Mode == "reject" {
		if err == nil {
			return errors.New("negative peer reached A2A")
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"rejected": true})
	}
	if err != nil {
		return err
	}
	for attempts := 0; attempts < 100; attempts++ {
		task, err = client.GetTask(ctx, &sdk.GetTaskRequest{Tenant: "allocation-probe", ID: task.ID})
		if err != nil {
			return err
		}
		if task.Status.State == sdk.TaskStateCompleted {
			break
		}
		if task.Status.State == sdk.TaskStateFailed {
			return errors.New("completion failed")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(10 * time.Millisecond):
		}
	}
	if task.Status.State != sdk.TaskStateCompleted || task.Status.Message == nil {
		return errors.New("missing terminal completion")
	}
	message := task.Status.Message
	if len(message.Parts) != 1 || message.Parts[0].MediaType != "application/vnd.contractor.worker-completion+json" {
		return errors.New("missing WorkerCompletion DataPart")
	}
	data, ok := message.Parts[0].Content.(sdk.Data)
	if !ok {
		return errors.New("expected structured completion")
	}
	encoded, err := json.Marshal(data.Value)
	if err != nil {
		return err
	}
	completion, err := contracts.DecodePrivateStrict[contracts.WorkerCompletion](encoded)
	if err != nil {
		return err
	}
	if completion.StateRevision != contracts.MaxExactInteger || completion.Result == nil || completion.Result.SubtaskID != "0" {
		return errors.New("WorkerCompletion or exact integer changed in transport")
	}
	if _, err = client.GetTask(ctx, &sdk.GetTaskRequest{Tenant: "foreign-allocation", ID: task.ID}); err == nil {
		return errors.New("foreign tenant could read Task")
	}
	task, err = send("block")
	if err != nil {
		return err
	}
	task, err = client.CancelTask(ctx, &sdk.CancelTaskRequest{Tenant: "allocation-probe", ID: task.ID})
	if err != nil {
		return err
	}
	// A2A cancellation delivery is not authoritative allocation shutdown.
	if task.ID == "" {
		return errors.New("CancelTask returned no Task identity")
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"completed": true, "cancellation_delivered": true, "tenant_rejected": true})
}

func serve(cfg config) error {
	tlsConfig, err := mtls.ControlPlaneServerConfig(mtls.Files{Certificate: cfg.Certificate, PrivateKey: cfg.Key, CA: cfg.CA})
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	defer listener.Close()
	var calls atomic.Int32
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(tlsConfig)), grpc.MaxRecvMsgSize(1024))
	server.RegisterService(&grpc.ServiceDesc{ServiceName: "probe.Control", HandlerType: (*interface{})(nil), Methods: []grpc.MethodDesc{{MethodName: "Register", Handler: func(_ any, ctx context.Context, decode func(any) error, _ grpc.UnaryServerInterceptor) (any, error) {
		var request wrapperspb.BytesValue
		if err := decode(&request); err != nil {
			return nil, err
		}
		p, ok := peer.FromContext(ctx)
		if !ok {
			return nil, errors.New("no peer")
		}
		info, ok := p.AuthInfo.(credentials.TLSInfo)
		if !ok {
			return nil, errors.New("no TLS")
		}
		id, err := mtls.RuntimeAgentIDFromConnection(&info.State)
		if err != nil {
			return nil, err
		}
		if id != cfg.Principal {
			return nil, errors.New("wrong Runtime principal")
		}
		calls.Add(1)
		if string(request.Value) == "lose-reply" {
			return nil, status.Error(codes.Unavailable, "probe mutation committed; reply unavailable")
		}
		return wrapperspb.Bytes([]byte("accepted")), nil
	}}}}, struct{}{})
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"port": listener.Addr().(*net.TCPAddr).Port}); err != nil {
		server.Stop()
		<-done
		return err
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
	server.Stop()
	if err := <-done; err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"calls": calls.Load()})
}

type principals struct{ calls atomic.Int32 }

func (p *principals) Register(_ context.Context, identity string) (agentprincipal.Principal, error) {
	p.calls.Add(1)
	return agentprincipal.Principal{RuntimeAgentID: identity, Labels: []string{}, LabelRevision: 1}, nil
}
func serveControl(cfg config) error {
	tlsConfig, err := mtls.ControlPlaneServerConfig(mtls.Files{Certificate: cfg.Certificate, PrivateKey: cfg.Key, CA: cfg.CA})
	if err != nil {
		return err
	}
	tlsConfig.NextProtos = []string{"h2", "http/1.1"}
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		return err
	}
	principals := &principals{}
	rpc, err := controlplane.NewGRPCHandler(registry, principals)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	defer listener.Close()
	server := &http.Server{TLSConfig: tlsConfig, ReadHeaderTimeout: 5 * time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cfg.Mode == "control-block-heartbeat" && r.URL.Path == runtimepb.ControlPlane_Heartbeat_FullMethodName {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]bool{"heartbeat": true})
			<-r.Context().Done()
			return
		}
		rpc.ServeHTTP(w, r)
	})}
	if err := http2.ConfigureServer(server, &http2.Server{MaxConcurrentStreams: 32}); err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(listener, tlsConfig)) }()
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{"port": listener.Addr().(*net.TCPAddr).Port}); err != nil {
		_ = server.Close()
		<-done
		return err
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
	_ = server.Close()
	err = <-done
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{"calls": principals.calls.Load()})
}
