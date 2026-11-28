//go:build e2e

package e2e

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sort"
	"sync"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"
)

// This fault boundary forwards real publication to the real Server/PG before
// dropping its acknowledgement. A later definite refusal tests isolation from
// target traffic; no production test hook or fake ArtifactStore is installed.
type sitemapPublicationProxy struct {
	server          *httptest.Server
	upstream        *url.URL
	clients         map[string]*http.Client
	faultIdentity   string
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
		client := newMTLSClient(t, caFile, identity)
		client.Transport.(*http.Transport).ForceAttemptHTTP2 = true
		proxy.clients[string(pair.Certificate[0])] = client
		if proxy.faultIdentity == "" {
			proxy.faultIdentity = string(pair.Certificate[0])
		}
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
	proxy.server.EnableHTTP2 = true
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
	// Only the first Runtime's publication is faulted. Other Runtime slots
	// share this advertised Artifact API and must retain independent collection.
	publication := r.Method == http.MethodPost && r.URL.Path == pb.Allocation_CollectSitemap_FullMethodName &&
		r.TLS != nil && len(r.TLS.PeerCertificates) > 0 && string(r.TLS.PeerCertificates[0].Raw) == p.faultIdentity
	var publicationKey string
	if publication {
		body, err := io.ReadAll(io.LimitReader(r.Body, (256<<10)+1024+6))
		if err != nil || len(body) < 5 || body[0] != 0 || int(binary.BigEndian.Uint32(body[1:5])) != len(body)-5 {
			http.Error(w, "invalid publication frame", 400)
			return
		}
		var input pb.SitemapCollectionRequest
		if proto.Unmarshal(body[5:], &input) != nil {
			http.Error(w, "invalid publication request", 400)
			return
		}
		publicationKey = input.GetIdempotencyKey()
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
	}
	p.mu.Lock()
	reject := publication && p.reject
	if publication {
		p.publicationKeys = append(p.publicationKeys, publicationKey)
	}
	if reject {
		p.rejected++
	}
	p.mu.Unlock()
	if reject {
		runtimewire.WriteError(w, runtimewire.Error(r.Context(), codes.FailedPrecondition, "artifact_history_protected", false))
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
		grpcStatus := response.Trailer.Get("Grpc-Status")
		if grpcStatus == "" {
			grpcStatus = response.Header.Get("Grpc-Status")
		}
		if grpcStatus == "0" {
			var result pb.SitemapMutationResponse
			if len(body) < 5 || body[0] != 0 || int(binary.BigEndian.Uint32(body[1:5])) != len(body)-5 || proto.Unmarshal(body[5:], &result) != nil || result.GetResult() == nil {
				http.Error(w, "invalid publication receipt", 502)
				return
			}
			p.mu.Lock()
			drop := !p.dropped
			p.dropped = true
			p.replayed = p.replayed || result.GetResult().GetReplayed()
			p.mu.Unlock()
			if drop {
				// HTTP/2 has no Hijacker. Abort this stream after the real commit.
				panic(http.ErrAbortHandler)
			}
		}
		response.Body = io.NopCloser(bytes.NewReader(body))
	}
	for name, values := range response.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, response.Body)
	if len(response.Trailer) > 0 {
		_ = http.NewResponseController(w).Flush()
	}
	for name, values := range response.Trailer {
		w.Header()[http.TrailerPrefix+name] = values
	}
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
// before the shared view is fetched. The subsequent HTTP body read still works
// when automatic publication is refused; the read serves committed map data.
func sitemapHTTPGatewayStage(stage domainGatewayStage) domainGatewayStage {
	stage.tools = append(stage.tools, "list_sitemap_entries", "read_sitemap_entry", "upsert_sitemap_entry", "snapshot_sitemap_context")
	sort.Strings(stage.tools)
	read := toolGatewayStep("list_sitemap_entries", fixedArguments(map[string]any{"refresh": true, "limit": 100}))
	read.validate = requireToolOutput("500")
	readManual := toolGatewayStep("read_sitemap_entry", func(request map[string]any) (map[string]any, error) {
		if _, err := unifiedSitemapSelection(request); err != nil {
			return nil, err
		}
		for _, result := range sitemapToolResponses(request, "list_sitemap_entries") {
			entries, _ := result["entries"].([]any)
			for _, raw := range entries {
				entry, _ := raw.(map[string]any)
				if entry["route"] == "/manual-canary" {
					return map[string]any{"entry_id": entry["id"]}, nil
				}
			}
		}
		return nil, fmt.Errorf("manual entry unavailable to editor")
	})
	annotate := toolGatewayStep("upsert_sitemap_entry", func(request map[string]any) (map[string]any, error) {
		for _, result := range sitemapToolResponses(request, "read_sitemap_entry") {
			entries, _ := result["entries"].([]any)
			for _, raw := range entries {
				entry, _ := raw.(map[string]any)
				value, _ := entry["value"].(map[string]any)
				if entry["kind"] == "member" && value["path"] == "/manual-canary" {
					return map[string]any{"member_ids": []any{entry["memberId"]}, "note": "unified-editor"}, nil
				}
			}
		}
		return nil, fmt.Errorf("manual member unavailable to editor")
	})
	metadata := toolGatewayStep("list_sitemap_entries", fixedArguments(map[string]any{"scope": "metadata", "limit": 100}))
	metadata.validate = func(request map[string]any) error {
		_, err := unifiedSitemapSelection(request)
		return err
	}
	snapshot := func(required bool) domainGatewayStep {
		return toolGatewayStep("snapshot_sitemap_context", func(request map[string]any) (map[string]any, error) {
			groups, err := unifiedSitemapSelection(request)
			return map[string]any{"group_ids": groups, "require_referenced_inputs": required}, err
		})
	}
	denied := snapshot(true)
	denied.validate = requireUnifiedSitemapMetadata
	retained := snapshot(false)
	retained.validate = requireToolOutput("artifact_not_found")
	following := stage.steps[3]
	following.validate = func(request map[string]any) error {
		for _, result := range sitemapToolResponses(request, "snapshot_sitemap_context") {
			artifact, _ := result["artifact"].(map[string]any)
			if result["status"] == "completed" && artifact["mediaType"] == "application/vnd.contractor.project-sitemap-context+json" {
				return nil
			}
		}
		return fmt.Errorf("shared sitemap snapshot was not retained")
	}
	steps := append([]domainGatewayStep{}, stage.steps[:3]...)
	steps = append(steps, read, readManual, annotate, metadata, denied, retained, following)
	stage.steps = append(steps, stage.steps[4:]...)
	return stage
}
