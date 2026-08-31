//go:build e2e

package uistack

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
)

const credentialRemoteKeyID = "b7a0c12fb3f64b41a1c6c91b9ced49cff0e66968df4e71da3d7834a889577ce1"

type credentialManagerFixture struct {
	server       *httptest.Server
	adminKey     string
	generatedKey string

	mu          sync.Mutex
	createCalls int
	deleteCalls int
	failures    []string
}

func newCredentialManagerFixture(adminKey, generatedKey string) *credentialManagerFixture {
	fixture := &credentialManagerFixture{adminKey: adminKey, generatedKey: generatedKey}
	fixture.server = httptest.NewServer(http.HandlerFunc(fixture.serveHTTP))
	return fixture
}

func (f *credentialManagerFixture) close() { f.server.Close() }
func (f *credentialManagerFixture) URL() string {
	return f.server.URL
}

func (f *credentialManagerFixture) snapshot() (creates, deletes int, failures []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.createCalls, f.deleteCalls, append([]string(nil), f.failures...)
}

func (f *credentialManagerFixture) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost ||
		(r.URL.Path != "/key/generate" && r.URL.Path != "/key/delete") {
		f.fail(w, http.StatusNotFound, "unsupported credential manager endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+f.adminKey ||
		r.Header.Get("Content-Type") != "application/json" ||
		r.Header.Get("Accept") != "application/json" {
		f.fail(w, http.StatusUnauthorized, "invalid credential manager headers")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		f.fail(w, http.StatusBadRequest, "invalid credential manager request")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if r.URL.Path == "/key/delete" {
		if aliases, ok := request["key_aliases"].([]any); ok && len(aliases) == 1 {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"message":"{'error': 'No keys found'}","type":"internal_server_error","param":"None","code":"404"}}`))
			return
		}
		keys, ok := request["keys"].([]any)
		if !ok || len(keys) != 1 || keys[0] != credentialRemoteKeyID {
			f.fail(w, http.StatusBadRequest, "delete did not use the generated token ID")
			return
		}
		f.mu.Lock()
		f.deleteCalls++
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"deleted_keys": []any{credentialRemoteKeyID}})
		return
	}
	alias, aliasOK := request["key_alias"].(string)
	models, modelsOK := request["models"].([]any)
	metadata, metadataOK := request["metadata"].(map[string]any)
	if !aliasOK || !strings.HasPrefix(alias, "contractor-") || !modelsOK || len(models) == 0 ||
		!metadataOK || metadata["contractorCredentialId"] != "ui-stack-key" {
		f.fail(w, http.StatusBadRequest, "generate request is not the exact Contractor policy")
		return
	}
	response := map[string]any{
		"key": f.generatedKey, "token": credentialRemoteKeyID,
		"token_id": credentialRemoteKeyID, "key_alias": alias,
		"models": models, "metadata": metadata,
		"allowed_routes": []any{"llm_api_routes"}, "blocked": false,
		"max_budget": request["max_budget"], "budget_duration": request["budget_duration"],
		"tpm_limit": request["tpm_limit"], "rpm_limit": request["rpm_limit"],
		"max_parallel_requests": request["max_parallel_requests"],
	}
	f.mu.Lock()
	f.createCalls++
	f.mu.Unlock()
	if err := json.NewEncoder(w).Encode(response); err != nil {
		panic(fmt.Sprintf("encode credential manager fixture response: %v", err))
	}
}

func (f *credentialManagerFixture) fail(w http.ResponseWriter, status int, message string) {
	f.mu.Lock()
	f.failures = append(f.failures, message)
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": message})
}
