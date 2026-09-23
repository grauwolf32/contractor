package publicclient

import (
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestClientAuthenticatesAndChecksAPIVersion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/workflows" || r.Header.Get("Authorization") != "Bearer secret" {
			t.Fatalf("request = %s %s auth=%q", r.Method, r.URL.Path, r.Header.Get("Authorization"))
		}
		w.Header().Set(APIVersionHeader, APIVersion)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[],"page":{"hasMore":false}}`))
	}))
	defer server.Close()

	client, err := New(Options{Server: server.URL, Token: "secret", Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.API.ListWorkflowsWithResponse(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if response.JSON200 == nil || len(response.JSON200.Items) != 0 {
		t.Fatalf("response = %+v", response)
	}
}

func TestClientRejectsMissingAPIVersion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()
	client, err := New(Options{Server: server.URL, Token: "secret", Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.API.ListWorkflowsWithResponse(t.Context(), nil)
	var compatibility *CompatibilityError
	if !errors.As(err, &compatibility) {
		t.Fatalf("error = %v", err)
	}
}

func TestClientConnectsWithAdditionalCA(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(APIVersionHeader, APIVersion)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[],"page":{"hasMore":false}}`))
	}))
	defer server.Close()
	caFile := filepath.Join(t.TempDir(), "ca.pem")
	encoded := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
	if err := os.WriteFile(caFile, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	client, err := New(Options{Server: server.URL, Token: "secret", CAFile: caFile, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.API.ListWorkflowsWithResponse(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckResponse(response, http.StatusOK); err != nil {
		t.Fatal(err)
	}
}

func TestClientRefusesRedirects(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(APIVersionHeader, APIVersion)
		w.Header().Set("Location", "/v1/workflows")
		w.WriteHeader(http.StatusFound)
	}))
	defer server.Close()
	client, err := New(Options{Server: server.URL, Token: "secret", Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.API.ListWorkflowsWithResponse(t.Context(), nil); err == nil {
		t.Fatal("redirect was followed")
	}
}

func TestRemoteHTTPRequiresExplicitOptIn(t *testing.T) {
	_, err := New(Options{
		Server: "http://192.0.2.1:8080", Token: "secret", Timeout: time.Second,
	})
	if err == nil {
		t.Fatal("remote HTTP was accepted")
	}
}

func TestTransportLeavesCallerRequestUnchanged(t *testing.T) {
	var seen http.Header
	transport := &checkedTransport{
		base: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			seen = request.Header.Clone()
			header := http.Header{}
			header.Set(APIVersionHeader, APIVersion)
			return &http.Response{StatusCode: http.StatusNoContent, Header: header, Body: http.NoBody, Request: request}, nil
		}),
		origin: "http://127.0.0.1:8080", token: "secret", userAgent: "contractor-test",
	}
	request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://127.0.0.1:8080/v1/workflows", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Caller", "kept")
	response, err := transport.RoundTrip(request)
	if err != nil {
		t.Fatal(err)
	}
	_ = response.Body.Close()
	if seen.Get("Authorization") != "Bearer secret" || seen.Get("User-Agent") != "contractor-test" ||
		seen.Get("X-Caller") != "kept" {
		t.Fatalf("sent headers = %v", seen)
	}
	if len(request.Header) != 1 || request.Header.Get("X-Caller") != "kept" {
		t.Fatalf("caller request headers were modified: %v", request.Header)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

// The Server accepts bearer tokens of 1 through 4096 bytes; a longer token
// must fail locally instead of producing an unexplained 401.
func TestTokenBoundMatchesServer(t *testing.T) {
	longest := strings.Repeat("t", 4096)
	if _, err := New(Options{Server: "http://127.0.0.1:8080", Token: longest, Timeout: time.Second}); err != nil {
		t.Fatalf("4096-byte token rejected: %v", err)
	}
	if _, err := New(Options{Server: "http://127.0.0.1:8080", Token: longest + "t", Timeout: time.Second}); err == nil {
		t.Fatal("4097-byte token accepted")
	}
	directory := t.TempDir()
	withLineEnding := filepath.Join(directory, "token")
	if err := os.WriteFile(withLineEnding, []byte(longest+"\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if token, err := ReadTokenFile(withLineEnding); err != nil || token != longest {
		t.Fatalf("token file with CRLF = %d bytes, %v", len(token), err)
	}
	tooLong := filepath.Join(directory, "long-token")
	if err := os.WriteFile(tooLong, []byte(longest+"t\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTokenFile(tooLong); err == nil {
		t.Fatal("4097-byte token file accepted")
	}
}

func TestQuoteETag(t *testing.T) {
	quoted, err := QuoteETag("rev_123")
	if err != nil || quoted != `"rev_123"` {
		t.Fatalf("quoted=%q err=%v", quoted, err)
	}
}
