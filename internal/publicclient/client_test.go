package publicclient

import (
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

func TestQuoteETag(t *testing.T) {
	quoted, err := QuoteETag("rev_123")
	if err != nil || quoted != `"rev_123"` {
		t.Fatalf("quoted=%q err=%v", quoted, err)
	}
}
