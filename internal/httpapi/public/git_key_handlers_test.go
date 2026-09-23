package public

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/credentials"
)

type recordingGitKeys struct{ replaced int }

func (*recordingGitKeys) Metadata(context.Context, string) (credentials.GitKeyMetadata, error) {
	return credentials.GitKeyMetadata{}, nil
}

func (k *recordingGitKeys) Replace(context.Context, string, []byte) (credentials.GitKeyMetadata, error) {
	k.replaced++
	return credentials.GitKeyMetadata{Configured: true}, nil
}

func (*recordingGitKeys) Delete(context.Context, string) error { return nil }

func newGitKeyFixture(t *testing.T, keys GitKeySettings) handlerFixture {
	t.Helper()
	return newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(d *Dependencies) { d.GitKeys = keys },
	)
}

func TestGitKeyRejectsUnsupportedMethods(t *testing.T) {
	keys := &recordingGitKeys{}
	fixture := newGitKeyFixture(t, keys)
	for _, method := range []string{http.MethodPost, http.MethodPatch} {
		request := authenticatedRequest(method, "/v1/settings/git-key", bytes.NewReader([]byte(`{"privateKey":"x"}`)))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusMethodNotAllowed ||
			!strings.Contains(response.Body.String(), `"method_not_allowed"`) || keys.replaced != 0 {
			t.Fatalf("%s Git key = %d: %s", method, response.Code, response.Body.String())
		}
	}
}

func TestGitKeyOversizedBodyIsRequestTooLarge(t *testing.T) {
	keys := &recordingGitKeys{}
	fixture := newGitKeyFixture(t, keys)
	body := []byte(`{"privateKey":"` + strings.Repeat("k", 64<<10) + `"}`)
	request := authenticatedRequest(http.MethodPut, "/v1/settings/git-key", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge ||
		!strings.Contains(response.Body.String(), `"request_too_large"`) || keys.replaced != 0 {
		t.Fatalf("oversized Git key = %d: %s", response.Code, response.Body.String())
	}
}

func TestGitKeyPublicAPIUsesAuthenticatedOwnerAndNeverReturnsSecrets(t *testing.T) {
	pool := isolatedPublicPool(t, t.Context())
	cipher, _ := credentials.NewTokenCipher(bytes.Repeat([]byte{1}, 32))
	keys := credentials.NewGitKeys(pool, cipher)
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) { d.GitKeys = keys })
	_, privateKey, _ := ed25519.GenerateKey(rand.Reader)
	encoded, _ := x509.MarshalPKCS8PrivateKey(privateKey)
	secret := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encoded})
	body, _ := json.Marshal(map[string]string{"privateKey": string(secret)})
	for _, method := range []string{http.MethodPut, http.MethodGet, http.MethodDelete, http.MethodGet} {
		var payload []byte
		if method == http.MethodPut {
			payload = body
		}
		request := authenticatedRequest(method, "/v1/settings/git-key", bytes.NewReader(payload))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		want := http.StatusOK
		if method == http.MethodDelete {
			want = http.StatusNoContent
		}
		if response.Code != want {
			t.Fatalf("%s: %d %s", method, response.Code, response.Body)
		}
		if bytes.Contains(response.Body.Bytes(), secret) || response.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("unsafe key response")
		}
		if method == http.MethodPut {
			own, err := keys.Metadata(t.Context(), "user-1")
			if err != nil || !own.Configured {
				t.Fatal("authenticated owner not used")
			}
			foreign, _ := keys.Metadata(t.Context(), "foreign")
			if foreign.Configured {
				t.Fatal("foreign key configured")
			}
		}
	}
	denied := httptest.NewRecorder()
	fixture.handler.ServeHTTP(denied, httptest.NewRequest(http.MethodPut, "/v1/settings/git-key", bytes.NewReader(body)))
	if denied.Code != http.StatusUnauthorized {
		t.Fatal("unauthenticated key mutation accepted")
	}
	if count, _ := keys.Count(t.Context()); count != 0 {
		t.Fatal("unauthenticated key persisted")
	}
}
