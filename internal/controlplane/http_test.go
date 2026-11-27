package controlplane

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestPrivateHeartbeatPathMustMatchBodyAndUnknownAgentReregisters(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler, _ := NewHTTPHandler(registry, HTTPOptions{})
	heartbeatBody, _ := json.Marshal(heartbeat("unknown", 1, 0))

	mismatch := httptest.NewRecorder()
	handler.ServeHTTP(mismatch, verifiedRequest(http.MethodPost, "/private/v1/agents/other/heartbeat", heartbeatBody))
	if mismatch.Code != http.StatusBadRequest {
		t.Fatalf("mismatched instance status = %d", mismatch.Code)
	}

	response := httptest.NewRecorder()
	handler.ServeHTTP(response, verifiedRequest(http.MethodPost, "/private/v1/agents/unknown/heartbeat", heartbeatBody))
	if response.Code != http.StatusOK {
		t.Fatalf("unknown instance heartbeat status = %d", response.Code)
	}
	var decoded control.HeartbeatResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || decoded.Action != control.ActionReregister || decoded.AckSeq != 1 {
		t.Fatalf("unknown instance response = (%+v, %v)", decoded, err)
	}
}

func verifiedRequest(method, target string, body []byte) *http.Request {
	request := httptest.NewRequest(method, target, bytes.NewReader(body))
	certificate := &x509.Certificate{RawSubjectPublicKeyInfo: []byte("test-runtime-key")}
	request.TLS = &tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{certificate},
		VerifiedChains:   [][]*x509.Certificate{{certificate}},
	}
	request.Header.Set("Content-Type", "application/json")
	return request
}

type testPrincipalRegistrar struct{}

type failingPrincipalRegistrar struct{ err error }

func (f failingPrincipalRegistrar) Register(context.Context, string) (agentprincipal.Principal, error) {
	return agentprincipal.Principal{}, f.err
}

func (testPrincipalRegistrar) Register(
	_ context.Context, runtimeAgentID string,
) (agentprincipal.Principal, error) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	return agentprincipal.Principal{
		RuntimeAgentID: runtimeAgentID, Labels: []string{},
		LabelRevision: 1, CreatedBy: "test", CreatedAt: now, UpdatedBy: "test", UpdatedAt: now,
	}, nil
}

type pkiFiles struct {
	controlPlane mtls.Files
	agent        mtls.Files
}

func generatePKI(t *testing.T, name string) pkiFiles {
	t.Helper()
	root := filepath.Join(t.TempDir(), name)
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	options := localpki.LeafOptions{
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}
	controlPlane, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: options})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent-1", options)
	if err != nil {
		t.Fatal(err)
	}
	return pkiFiles{
		controlPlane: mtls.Files{Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate},
		agent:        mtls.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate},
	}
}
