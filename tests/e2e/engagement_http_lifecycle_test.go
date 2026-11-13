//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
)

// Shorten only the Runtime's confirmed control allowance. This is a fixture
// clock bound, never extra target authority; ordinary allocation authority remains unchanged.
type engagementLeaseTimingTransport struct{ next http.RoundTripper }

func (t engagementLeaseTimingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	next := t.next
	if next == nil {
		next = http.DefaultTransport
	}
	response, err := next.RoundTrip(r)
	if err != nil || !strings.HasSuffix(r.URL.Path, "/register") || response.StatusCode != 200 {
		return response, err
	}
	return shortConfirmedControlLease(response)
}

func shortConfirmedControlLease(response *http.Response) (*http.Response, error) {
	body, err := io.ReadAll(io.LimitReader(response.Body, 4096))
	_ = response.Body.Close()
	if err != nil {
		return nil, err
	}
	var reply control.AgentRegistrationResponse
	if err = json.Unmarshal(body, &reply); err != nil {
		return nil, err
	}
	reply.HeartbeatIntervalSeconds = 1
	reply.ConfirmedLeaseSeconds = 3
	body, err = json.Marshal(reply)
	if err != nil {
		return nil, err
	}
	response.Body = io.NopCloser(bytes.NewReader(body))
	response.ContentLength = int64(len(body))
	response.Header.Del("Content-Length")
	return response, nil
}

func TestEngagementHTTPLifecycleAcrossProcesses(t *testing.T) {
	for _, mode := range []string{"cancel_joins", "lease_loss_joins", "server_restart", "runtime_restart_joins"} {
		t.Run(mode, func(t *testing.T) {
			secure := newEngagementTargetFixture(t)
			target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
			entered, closed := make(chan struct{}), make(chan struct{})
			var once, closeOnce sync.Once
			target.setHandler(func(w http.ResponseWriter, r *http.Request) {
				once.Do(func() { close(entered) })
				<-r.Context().Done()
				closeOnce.Do(func() { close(closed) })
			})
			proxyMode := "pass"
			if mode == "lease_loss_joins" {
				proxyMode = "lease"
			}
			h := startEngagementProcessHarness(t, secure.caFile, proxyMode)
			pool := engagementPool(t, h)
			audit := startEngagementMatrixAudit(t, h, []string{target.server.URL}, []string{target.server.URL + "/stall"}, []engagementcontract.Route{engagementRoute(target.server.URL, "/")}, nil, 1024)
			select {
			case <-entered:
			case <-h.ctx.Done():
				t.Fatal("target request did not start")
			}
			switch mode {
			case "cancel_joins":
				var revision uint64
				if err := pool.QueryRow(h.ctx, `SELECT revision FROM audits WHERE audit_id=$1`, audit).Scan(&revision); err != nil {
					t.Fatal(err)
				}
				engagementAuditPOST(t, h, "/v1/audits/"+audit+"/cancel", nil, revision, http.StatusAccepted, nil)
			case "lease_loss_joins":
				h.proxy.faultMu.Lock()
				h.proxy.blockHeartbeats = true
				h.proxy.faultMu.Unlock()
			case "server_restart":
				if err := h.server.cmd.Process.Kill(); err != nil {
					t.Fatal(err)
				}
				select {
				case <-h.server.done:
				case <-h.ctx.Done():
					t.Fatal("old Server did not exit")
				}
				h.server = h.startServer()
				waitForHTTP(t, h.ctx, h.server, h.client, h.baseURL+"/readyz", http.StatusOK)
			case "runtime_restart_joins":
				if err := h.runtimeProcess.cmd.Process.Kill(); err != nil {
					t.Fatal(err)
				}
				select {
				case <-h.runtimeProcess.done:
				case <-h.ctx.Done():
					t.Fatal("old Runtime did not exit")
				}
				h.runtimeProcess = h.startRuntime()
				waitForProcessLog(t, h.ctx, h.runtimeProcess, "runtime agent registered")
			}
			select {
			case <-closed:
			case <-h.ctx.Done():
				t.Fatal("allocation cleanup did not close target connection")
			}
			if len(target.requests()) != 1 {
				t.Fatal("failed in-flight target request was sent again")
			}
			h.proxy.verify(t)
		})
	}
}
