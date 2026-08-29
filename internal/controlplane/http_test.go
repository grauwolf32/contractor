package controlplane

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestPrivateHTTPRequiresVerifiedMTLSAndStrictBody(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler, err := NewHTTPHandler(registry, HTTPOptions{
		NewRequestID: func() (string, error) { return "control-request-fixed", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	registration := testRegistration("agent-http")
	body, _ := json.Marshal(registration)

	unauthenticated := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/private/v1/agents/register", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(unauthenticated, request)
	if unauthenticated.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d", unauthenticated.Code)
	}
	var unauthenticatedError privateErrorResponse
	if err := json.Unmarshal(unauthenticated.Body.Bytes(), &unauthenticatedError); err != nil ||
		unauthenticated.Header().Get("X-Request-ID") != "control-request-fixed" ||
		unauthenticatedError.RequestID != "control-request-fixed" {
		t.Fatalf("unauthenticated correlation = headers:%v body:%+v error:%v",
			unauthenticated.Header(), unauthenticatedError, err)
	}

	trusted := httptest.NewRecorder()
	request = verifiedRequest(http.MethodPost, "/private/v1/agents/register", body)
	handler.ServeHTTP(trusted, request)
	if trusted.Code != http.StatusOK {
		t.Fatalf("registration status = %d, body %s", trusted.Code, trusted.Body.String())
	}
	var response contracts.AgentRegistrationResponse
	if err := json.Unmarshal(trusted.Body.Bytes(), &response); err != nil || response.HeartbeatIntervalSeconds != 10 || response.ConfirmedLeaseSeconds != 60 {
		t.Fatalf("registration response = (%+v, %v)", response, err)
	}

	invalidBody := append(bytes.TrimSuffix(body, []byte("}")), []byte(`,"unknown":true}`)...)
	invalid := httptest.NewRecorder()
	handler.ServeHTTP(invalid, verifiedRequest(http.MethodPost, "/private/v1/agents/register", invalidBody))
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("unknown field status = %d, body %s", invalid.Code, invalid.Body.String())
	}
}

func TestPrivateHeartbeatPathMustMatchBodyAndUnknownAgentReregisters(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler, _ := NewHTTPHandler(registry)
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
	var decoded contracts.HeartbeatResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || decoded.Action != contracts.ActionReregister || decoded.AckSeq != 1 {
		t.Fatalf("unknown instance response = (%+v, %v)", decoded, err)
	}
}

func TestPrivateRegistryMTLSRejectsForeignCA(t *testing.T) {
	trusted := generatePKI(t, "trusted")
	foreign := generatePKI(t, "foreign")
	registry := newTestRegistry(t, newTestClock())
	handler, _ := NewHTTPHandler(registry)
	serverTLS, err := mtls.ControlPlaneServerConfig(trusted.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(handler)
	server.Config.ErrorLog = log.New(io.Discard, "", 0)
	server.TLS = serverTLS
	server.StartTLS()
	defer server.Close()

	registration := testRegistration("trusted-agent")
	body, _ := json.Marshal(registration)
	trustedClientTLS, err := mtls.ControlPlaneClientConfig(trusted.agent, "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	trustedClient := &http.Client{Transport: &http.Transport{TLSClientConfig: trustedClientTLS}, Timeout: 3 * time.Second}
	response, err := postJSON(trustedClient, server.URL+"/private/v1/agents/register", body)
	if err != nil {
		t.Fatal(err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("trusted mTLS status = %d", response.StatusCode)
	}

	foreignIdentity := foreign.agent
	foreignIdentity.CA = trusted.controlPlane.CA
	foreignClientTLS, err := mtls.ControlPlaneClientConfig(foreignIdentity, "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	foreignClient := &http.Client{Transport: &http.Transport{TLSClientConfig: foreignClientTLS}, Timeout: 3 * time.Second}
	foreignRegistration := testRegistration("foreign-agent")
	foreignBody, _ := json.Marshal(foreignRegistration)
	if response, err := postJSON(foreignClient, server.URL+"/private/v1/agents/register", foreignBody); err == nil {
		_ = response.Body.Close()
		t.Fatal("foreign-CA HTTP client reached the private handler")
	}
	if _, err := registry.GetAgent("foreign-agent"); err == nil {
		t.Fatal("foreign-CA request mutated the registry")
	}
}

func verifiedRequest(method, target string, body []byte) *http.Request {
	request := httptest.NewRequest(method, target, bytes.NewReader(body))
	certificate := &x509.Certificate{}
	request.TLS = &tls.ConnectionState{
		PeerCertificates: []*x509.Certificate{certificate},
		VerifiedChains:   [][]*x509.Certificate{{certificate}},
	}
	request.Header.Set("Content-Type", "application/json")
	return request
}

func postJSON(client *http.Client, target string, body []byte) (*http.Response, error) {
	request, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	return client.Do(request)
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
