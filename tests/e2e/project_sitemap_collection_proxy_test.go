//go:build e2e

package e2e

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/localpki"
)

// This fault boundary forwards real publication to the real Server/PG before
// dropping its acknowledgement. A later definite refusal tests isolation from
// target traffic; no production test hook or fake ArtifactStore is installed.
type sitemapPublicationProxy struct {
	server          *httptest.Server
	upstream        *url.URL
	clients         map[string]*http.Client
	mu              sync.Mutex
	reject          bool
	dropped         bool
	replayed        bool
	publicationKeys []string
	rejected        int
}

func newSitemapPublicationProxy(t *testing.T, target, caFile string, runtimeIdentities []localpki.Paths, controlIdentity localpki.Paths) *sitemapPublicationProxy {
	t.Helper()
	upstream, err := url.Parse(target)
	if err != nil {
		t.Fatal(err)
	}
	proxy := &sitemapPublicationProxy{upstream: upstream, clients: map[string]*http.Client{}}
	for _, identity := range runtimeIdentities {
		pair, err := tls.LoadX509KeyPair(identity.Certificate, identity.PrivateKey)
		if err != nil {
			t.Fatal(err)
		}
		proxy.clients[string(pair.Certificate[0])] = newMTLSClient(t, caFile, identity)
	}
	certificate, err := tls.LoadX509KeyPair(controlIdentity.Certificate, controlIdentity.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	ca, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(ca) {
		t.Fatal("invalid private proxy CA")
	}
	proxy.server = httptest.NewUnstartedServer(http.HandlerFunc(proxy.serveHTTP))
	proxy.server.TLS = &tls.Config{MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{certificate}, ClientAuth: tls.RequireAndVerifyClientCert, ClientCAs: roots}
	proxy.server.StartTLS()
	t.Cleanup(func() {
		proxy.server.Close()
		for _, client := range proxy.clients {
			client.CloseIdleConnections()
		}
	})
	return proxy
}

func (p *sitemapPublicationProxy) rejectPublications() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reject = true
}

func (p *sitemapPublicationProxy) serveHTTP(w http.ResponseWriter, r *http.Request) {
	publication := r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/sitemap/observations")
	p.mu.Lock()
	reject := publication && p.reject
	if publication {
		p.publicationKeys = append(p.publicationKeys, r.Header.Get("Idempotency-Key"))
	}
	if reject {
		p.rejected++
	}
	p.mu.Unlock()
	if reject {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_ = json.NewEncoder(w).Encode(map[string]any{"code": "artifact_history_protected", "message": "publication refused", "retryable": false, "requestId": "sitemap-fault"})
		return
	}
	forwarded := r.Clone(r.Context())
	forwarded.URL.Scheme, forwarded.URL.Host = p.upstream.Scheme, p.upstream.Host
	forwarded.RequestURI = ""
	forwarded.Host = p.upstream.Host
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		http.Error(w, "private identity unavailable", 403)
		return
	}
	client := p.clients[string(r.TLS.PeerCertificates[0].Raw)]
	if client == nil {
		http.Error(w, "private identity unavailable", 403)
		return
	}
	response, err := client.Do(forwarded)
	if err != nil {
		http.Error(w, "private upstream unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if publication && response.StatusCode == http.StatusOK {
		body, err := io.ReadAll(io.LimitReader(response.Body, 32<<10))
		if err != nil {
			http.Error(w, "private reply unavailable", 502)
			return
		}
		p.mu.Lock()
		drop := !p.dropped
		p.dropped = true
		var result struct {
			Result struct {
				Replayed bool `json:"replayed"`
			} `json:"result"`
		}
		if json.Unmarshal(body, &result) == nil && result.Result.Replayed {
			p.replayed = true
		}
		p.mu.Unlock()
		if drop {
			connection, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = connection.Close()
			}
			return
		}
		for name, values := range response.Header {
			for _, value := range values {
				w.Header().Add(name, value)
			}
		}
		w.WriteHeader(response.StatusCode)
		_, _ = w.Write(body)
		return
	}
	for name, values := range response.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, response.Body)
}

func (p *sitemapPublicationProxy) verify(t *testing.T) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.dropped || !p.replayed || p.rejected != 1 || len(p.publicationKeys) != 3 || p.publicationKeys[0] == "" || p.publicationKeys[0] != p.publicationKeys[1] {
		t.Fatalf("publication reconciliation: dropped=%t replayed=%t rejected=%d requests=%d", p.dropped, p.replayed, p.rejected, len(p.publicationKeys))
	}
}

// A live reader proves that its own pending HTTP observation is published
// before the safe view is fetched. The subsequent HTTP body read still works
// when automatic publication is refused; the read serves committed map data.
func sitemapHTTPGatewayStage(stage domainGatewayStage) domainGatewayStage {
	stage.tools = append(stage.tools, "list_sitemap_entries")
	sort.Strings(stage.tools)
	read := toolGatewayStep("list_sitemap_entries", fixedArguments(map[string]any{"refresh": true, "limit": 100}))
	read.validate = requireToolOutput("500")
	following := stage.steps[3]
	following.validate = requireToolOutput("/failure")
	steps := append([]domainGatewayStep{}, stage.steps[:3]...)
	steps = append(steps, read, following)
	stage.steps = append(steps, stage.steps[4:]...)
	return stage
}
