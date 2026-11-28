//go:build e2e

package e2e

import (
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/localpki"
)

// The fixture observes that ordinary private traffic never uses a request
// journal RPC. It can interrupt heartbeats to exercise the normal lease path.
type engagementActionProxy struct {
	*sitemapPublicationProxy
	faultMu          sync.Mutex
	blockHeartbeats  bool
	obsoleteRequests int
}

func newEngagementActionProxy(t *testing.T, target, ca string, agent, control localpki.Paths, mode string) *engagementActionProxy {
	base := newSitemapPublicationProxy(t, target, ca, []localpki.Paths{agent}, control)
	p := &engagementActionProxy{sitemapPublicationProxy: base}
	if mode == "lease" {
		for _, client := range p.clients {
			client.Transport = engagementLeaseTimingTransport{client.Transport}
		}
	}
	base.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.faultMu.Lock()
		blocked := p.blockHeartbeats && r.URL.Path == runtimepb.ControlPlane_Heartbeat_FullMethodName
		obsolete := strings.HasSuffix(r.URL.Path, "/engagement-actions")
		if obsolete {
			p.obsoleteRequests++
		}
		p.faultMu.Unlock()
		if blocked {
			http.Error(w, "fixture dropped heartbeat", http.StatusServiceUnavailable)
			return
		}
		if obsolete {
			http.Error(w, "removed request journal RPC", http.StatusNotFound)
			return
		}
		p.sitemapPublicationProxy.serveHTTP(w, r)
	})
	return p
}
func (p *engagementActionProxy) verify(t *testing.T) {
	t.Helper()
	p.faultMu.Lock()
	defer p.faultMu.Unlock()
	if p.obsoleteRequests != 0 {
		t.Fatalf("Runtime invoked removed per-request Server protocol %d times", p.obsoleteRequests)
	}
}
