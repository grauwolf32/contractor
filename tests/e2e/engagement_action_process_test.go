//go:build e2e

package e2e

import (
	"crypto/tls"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/localpki"
)

type engagementTargetFixture struct {
	httpURL, httpsURL, caFile                                      string
	httpRequests, tlsRequests, requestBytes, unexpectedCredentials atomic.Int64
}
type engagementCountListener struct {
	net.Listener
	bytes *atomic.Int64
}
type engagementCountConn struct {
	net.Conn
	bytes *atomic.Int64
}

func (l engagementCountListener) Accept() (net.Conn, error) {
	c, e := l.Listener.Accept()
	if e != nil {
		return nil, e
	}
	return engagementCountConn{c, l.bytes}, nil
}
func (c engagementCountConn) Read(p []byte) (int, error) {
	n, e := c.Conn.Read(p)
	c.bytes.Add(int64(n))
	return n, e
}

func newEngagementTargetFixture(t *testing.T) *engagementTargetFixture {
	t.Helper()
	ip := engagementFixtureBindIP(t)
	var err error
	f := &engagementTargetFixture{}
	root := filepath.Join(t.TempDir(), "target-pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: localpki.LeafOptions{IPAddresses: []net.IP{ip}}, URI: "urn:contractor:control-plane:engagement-target-fixture"})
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := tls.LoadX509KeyPair(leaf.Certificate, leaf.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	f.caFile = ca.Certificate
	secure := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.tlsRequests.Add(1)
		if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
			f.unexpectedCredentials.Add(1)
		}
		w.Header().Set("X-Fixture-Secret", "response-header-canary")
		_, _ = w.Write([]byte("response-body-canary"))
	}))
	_ = secure.Listener.Close()
	secure.Listener, err = net.Listen("tcp", net.JoinHostPort(ip.String(), "0"))
	if err != nil {
		t.Fatal(err)
	}
	secure.TLS = &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{certificate}}
	secure.StartTLS()
	t.Cleanup(secure.Close)
	f.httpsURL = secure.URL
	listener, err := net.Listen("tcp", net.JoinHostPort(ip.String(), "0"))
	if err != nil {
		t.Fatal(err)
	}
	plain := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.httpRequests.Add(1)
		if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" {
			f.unexpectedCredentials.Add(1)
		}
		if r.URL.Path == "/redirect" {
			w.Header().Set("Location", f.httpsURL+"/tls?redirect-query-canary=1")
			w.WriteHeader(http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("response-body-canary"))
	}))
	_ = plain.Listener.Close()
	plain.Listener = engagementCountListener{listener, &f.requestBytes}
	plain.Start()
	t.Cleanup(plain.Close)
	f.httpURL = plain.URL
	return f
}

func engagementFixtureBindIP(t *testing.T) net.IP {
	t.Helper()
	var ip net.IP
	requested := os.Getenv("CONTRACTOR_HTTP_TEST_BIND_ADDRESS")
	interfaces, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	for _, address := range interfaces {
		candidate, _, err := net.ParseCIDR(address.String())
		if err == nil && candidate.To4() != nil && candidate.IsPrivate() && !candidate.IsLoopback() && (requested == "" || candidate.String() == requested) {
			ip = candidate.To4()
			break
		}
	}
	if ip == nil {
		t.Fatal("engagement process gate requires an actual RFC1918 interface; loopback policy must remain enforced")
	}
	return ip
}
