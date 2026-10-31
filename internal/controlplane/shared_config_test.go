package controlplane

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestStartupConfigRequiresMTLSWithoutRegisteringPrincipal(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	handler, err := NewHTTPHandler(registry, HTTPOptions{Principals: failingPrincipalRegistrar{err: ErrInvalidRequest}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		method, path  string
		body          []byte
		authenticated bool
		status        int
	}{
		{"GET", "/private/v1/config", nil, false, 401},
		{"GET", "/private/v1/config", nil, true, 200},
		{"GET", "/private/v1/config?x=1", nil, true, 400},
		{"GET", "/private/v1/config", []byte(`{}`), true, 400},
		{"POST", "/private/v1/config", nil, true, 405},
		{"HEAD", "/private/v1/config", nil, true, 405},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			request := httptest.NewRequest(tc.method, tc.path, nil)
			if tc.authenticated {
				request = verifiedRequest(tc.method, tc.path, tc.body)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("status %d body %s", response.Code, response.Body)
			}
			if tc.status == http.StatusOK {
				expected, err := control.DefaultSharedConfig()
				if err != nil {
					t.Fatal(err)
				}
				got, err := contracts.DecodePrivateStrict[control.SharedConfig](response.Body.Bytes())
				if err != nil || got != expected {
					t.Fatalf("config = %+v, %v", got, err)
				}
				if response.Header().Get("Cache-Control") != "no-store" {
					t.Fatal("config may be cached")
				}
			}
		})
	}
	if len(registry.agents) != 0 {
		t.Fatal("config registered an agent")
	}
}
