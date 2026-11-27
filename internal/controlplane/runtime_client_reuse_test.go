package controlplane

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/mtlstest"
	"google.golang.org/grpc"
)

// TestRuntimeControlClientReusesOneVerifiedConnectionUntilRelease drives the
// production mTLS client: ten lifecycle calls share one verified connection,
// Release closes it, and the next connection is verified again, rejecting a
// certificate of another key and accepting a renewal of the expected one.
func TestRuntimeControlClientReusesOneVerifiedConnectionUntilRelease(t *testing.T) {
	root := filepath.Join(t.TempDir(), "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlane, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	other, err := generator.IssueAgent(root, "other-key", leaf)
	if err != nil {
		t.Fatal(err)
	}
	var presented atomic.Pointer[tls.Certificate]
	present := func(paths localpki.Paths) {
		certificate, loadErr := tls.LoadX509KeyPair(paths.Certificate, paths.PrivateKey)
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		presented.Store(&certificate)
	}
	present(agent)
	agentTLS, err := mtlstest.AgentServerConfig(mtlstest.Files{
		Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate,
	})
	if err != nil {
		t.Fatal(err)
	}
	var handshakes, requests atomic.Int64
	verifyClient := agentTLS.VerifyConnection
	agentTLS.VerifyConnection = func(state tls.ConnectionState) error {
		if verifyErr := verifyClient(state); verifyErr != nil {
			return verifyErr
		}
		handshakes.Add(1)
		return nil
	}
	var accepted, closed atomic.Int64
	rpc := grpc.NewServer()
	defer rpc.Stop()
	fixture := &runtimeFixture{documents: http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requests.Add(1)
		_, _ = io.Copy(io.Discard, request.Body)
		if stringsHasSuffix(request.URL.Path, "/release") {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(reporting.AllocationFinalResponse{
			APIVersion: contracts.APIVersion, Report: testExecutionReport("allocation_reuse"),
		})
	})}
	pb.RegisterRuntimeLifecycleServer(rpc, fixture)
	server := httptest.NewUnstartedServer(rpc)
	agentTLS.NextProtos = []string{"h2"}
	server.TLS = &tls.Config{MinVersion: tls.VersionTLS13, GetConfigForClient: func(*tls.ClientHelloInfo) (*tls.Config, error) {
		current := agentTLS.Clone()
		current.Certificates = []tls.Certificate{*presented.Load()}
		return current, nil
	}}
	server.EnableHTTP2 = true
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			accepted.Add(1)
		case http.StateClosed:
			closed.Add(1)
		}
	}
	server.StartTLS()
	fixture.origin = server.URL
	defer server.Close()

	client, err := NewMTLSRuntimeControlClient(mtls.Files{
		Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate,
	}, 2*time.Second, &projectionGrantReader{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	reservation := testReservation(
		"allocation_reuse", "builder", server.URL, server.URL, testTemplate(t), time.Now().Add(time.Minute),
	)
	reservation.Grant.RuntimeAgentID = runtimePrincipalFromCertificate(t, agent.Certificate)
	finalize := func() error {
		_, finalizeErr := client.Finalize(context.Background(), reservation, "finalization_reuse", time.Now().Add(time.Minute))
		return finalizeErr
	}
	for index := range 9 {
		if err := finalize(); err != nil {
			t.Fatalf("finalize %d: %v", index, err)
		}
	}
	if err := client.Release(context.Background(), reservation); err != nil {
		t.Fatal(err)
	}
	if handshakes.Load() != 1 || requests.Load() != 10 {
		t.Fatalf("handshakes=%d requests=%d, want ten control calls on one verified connection",
			handshakes.Load(), requests.Load())
	}
	awaitServerClose(t, &accepted, &closed, "Release did not close the released principal's connection")

	present(other)
	if err := finalize(); err == nil || requests.Load() != 10 {
		t.Fatalf("reconnect to another key = %v, requests=%d; want rejection before request bytes", err, requests.Load())
	}
	renewed, err := generator.RenewAgent(root, "agent")
	if err != nil {
		t.Fatal(err)
	}
	present(renewed)
	if err := finalize(); err != nil || handshakes.Load() != 2 || requests.Load() != 11 {
		t.Fatalf("renewed certificate = %v, handshakes=%d requests=%d", err, handshakes.Load(), requests.Load())
	}

	client.Close()
	awaitServerClose(t, &accepted, &closed, "Close did not close the client's connections")
	err = finalize()
	if !errors.Is(err, mtls.ErrRuntimeChannelsClosed) || requests.Load() != 11 {
		t.Fatalf("call after Close = %v, requests=%d", err, requests.Load())
	}
}

// awaitServerClose waits until the peer observed every accepted connection
// closed. The deadline only bounds a failing test.
func awaitServerClose(t *testing.T, accepted, closed *atomic.Int64, failure string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for closed.Load() != accepted.Load() {
		if time.Now().After(deadline) {
			t.Fatalf("%s: accepted=%d closed=%d", failure, accepted.Load(), closed.Load())
		}
		time.Sleep(time.Millisecond)
	}
}
