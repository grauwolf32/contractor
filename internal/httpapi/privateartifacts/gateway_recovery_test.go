package privateartifacts

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
)

type recordingRecovery struct{ requests []gatewayrecovery.Request }

func (r *recordingRecovery) Update(
	_ context.Context, _ string, request gatewayrecovery.Request,
) (gatewayrecovery.Decision, error) {
	r.requests = append(r.requests, request)
	return gatewayrecovery.Decision{Allowed: true}, nil
}

func newRecoveryTestHandler(recovery gatewayRecoveryUpdater) http.Handler {
	current := &handler{
		dependencies: Dependencies{
			Registry:  &fakeRegistry{grant: testGrant("run-a")},
			Artifacts: artifacts.NewService(newMemoryRepository()),
		},
		recovery: recovery,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /private/v1/allocations/{allocationID}/gateway-recovery", current.gatewayRecovery)
	return current.requireMTLS(mux)
}

func TestGatewayRecoveryUsesPackageJSONContract(t *testing.T) {
	const body = `{"model":"model","requestId":"request-1","action":"acquire"}`
	for _, test := range []struct {
		name        string
		contentType string
		status      int
	}{
		{"json", "application/json", http.StatusOK},
		{"json with parameters", "application/json; charset=utf-8", http.StatusBadRequest},
		{"text", "text/plain", http.StatusBadRequest},
		{"missing", "", http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			recovery := &recordingRecovery{}
			request := trustedRequest(
				http.MethodPost, "/private/v1/allocations/allocation-1/gateway-recovery", bytes.NewReader([]byte(body)),
			)
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			response := httptest.NewRecorder()
			newRecoveryTestHandler(recovery).ServeHTTP(response, request)
			if response.Code != test.status || response.Header().Get("Content-Type") != "application/json" ||
				response.Header().Get("Cache-Control") != "no-store" ||
				response.Header().Get("Content-Length") != strconv.Itoa(response.Body.Len()) {
				t.Fatalf("recovery = %d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
			}
			if test.status != http.StatusOK {
				if len(recovery.requests) != 0 {
					t.Fatalf("rejected request reached recovery: %+v", recovery.requests)
				}
				return
			}
			var decision gatewayrecovery.Decision
			if err := json.Unmarshal(response.Body.Bytes(), &decision); err != nil || !decision.Allowed ||
				len(recovery.requests) != 1 || recovery.requests[0].RequestID != "request-1" {
				t.Fatalf("decision = %+v (%v), requests %+v", decision, err, recovery.requests)
			}
		})
	}
}

// A write-fenced allocation belongs to an ending Stage. It may still deliver
// the terminal update that relinquishes its probe, but it can neither start a
// model call nor report a failure that would make its Run wait.
func TestGatewayRecoveryFencedAllocationDeliversOnlyTerminalUpdates(t *testing.T) {
	for _, test := range []struct {
		action string
		status int
	}{
		{"succeeded", http.StatusOK},
		{"finished", http.StatusOK},
		{"released", http.StatusOK},
		{"acquire", http.StatusConflict},
		{"failed", http.StatusConflict},
	} {
		t.Run(test.action, func(t *testing.T) {
			grant := testGrant("run-a")
			grant.WriteFenced = true
			recovery := &recordingRecovery{}
			current := &handler{
				dependencies: Dependencies{
					Registry:  &fakeRegistry{grant: grant},
					Artifacts: artifacts.NewService(newMemoryRepository()),
				},
				recovery: recovery,
			}
			mux := http.NewServeMux()
			mux.HandleFunc("POST /private/v1/allocations/{allocationID}/gateway-recovery", current.gatewayRecovery)
			body := `{"model":"model","requestId":"request-1","action":"` + test.action + `"`
			if test.action == "failed" {
				body += `,"code":"model_unavailable"`
			}
			request := trustedRequest(
				http.MethodPost, "/private/v1/allocations/allocation-1/gateway-recovery",
				bytes.NewReader([]byte(body+`}`)),
			)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			current.requireMTLS(mux).ServeHTTP(response, request)
			delivered := len(recovery.requests)
			if response.Code != test.status || (test.status == http.StatusOK) != (delivered == 1) {
				t.Fatalf("fenced %s = %d body=%s, delivered %d", test.action, response.Code, response.Body.String(), delivered)
			}
		})
	}
}
