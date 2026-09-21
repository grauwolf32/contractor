package auth

import (
	"errors"
	"strings"
	"testing"
)

func TestNewPeerPolicyAcceptsAddressesAndPrefixesOnly(t *testing.T) {
	policy, err := NewPeerPolicy([]string{"10.0.0.0/8", "192.0.2.7", "2001:db8::/32", "2001:db8:1::1"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(policy.Values(), " "); got != "10.0.0.0/8 192.0.2.7/32 2001:db8:1::1/128 2001:db8::/32" {
		t.Fatalf("canonical values = %q", got)
	}
	for _, invalid := range [][]string{
		{""}, {" 10.0.0.1"}, {"proxy.example"}, {"10.0.0.1:8080"}, {"10.0.0.0/33"},
		{"0.0.0.0/0"}, {"::/0"}, {"fe80::1%eth0"}, {"::ffff:10.0.0.1"},
		{"10.0.0.1", "10.0.0.1/32"},
		make([]string, maximumTrustedProxies+1),
	} {
		if _, err := NewPeerPolicy(invalid); !errors.Is(err, ErrInvalidTrustedProxy) {
			t.Errorf("NewPeerPolicy(%q) accepted: %v", invalid, err)
		}
	}
	empty, err := NewPeerPolicy(nil)
	if err != nil || len(empty.Values()) != 0 {
		t.Fatalf("empty policy = %v, %v", empty.Values(), err)
	}
}

func TestClientIPIgnoresForwardedHeadersWithoutTrustedProxies(t *testing.T) {
	var policy PeerPolicy
	got, err := policy.ClientIP("198.51.100.4:51000", []string{"203.0.113.9"})
	if err != nil || got != "198.51.100.4" {
		t.Fatalf("ClientIP = %q, %v", got, err)
	}
	if _, err := policy.ClientIP("not-an-address", nil); err == nil {
		t.Fatal("malformed socket peer was accepted")
	}
}

func TestClientIPWalksForwardedChainFromTrustedProxies(t *testing.T) {
	policy, err := NewPeerPolicy([]string{"10.0.0.0/8", "2001:db8::/32"})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name      string
		remote    string
		forwarded []string
		want      string
	}{
		{"untrusted peer keeps its own address", "198.51.100.4:51000", []string{"203.0.113.9"}, "198.51.100.4"},
		{"trusted proxy without chain", "10.1.2.3:51000", nil, "10.1.2.3"},
		{"trusted proxy with empty chain", "10.1.2.3:51000", []string{""}, "10.1.2.3"},
		{"single hop", "10.1.2.3:51000", []string{"203.0.113.9"}, "203.0.113.9"},
		{"client spoof is skipped", "10.1.2.3:51000", []string{"192.0.2.250, 203.0.113.9"}, "203.0.113.9"},
		{"nested trusted proxies", "10.1.2.3:51000", []string{"203.0.113.9, 10.9.9.9"}, "203.0.113.9"},
		{"multiple header lines", "10.1.2.3:51000", []string{"192.0.2.250", "203.0.113.9, 10.9.9.9"}, "203.0.113.9"},
		{"all hops trusted uses outermost", "10.1.2.3:51000", []string{"10.5.5.5, 10.9.9.9"}, "10.5.5.5"},
		{"bracketed and ported hops", "10.1.2.3:51000", []string{"[2001:db8:ffff::1]:443"}, "2001:db8:ffff::1"},
		{"foreign ipv6 client", "10.1.2.3:51000", []string{"2001:4860::8888"}, "2001:4860::8888"},
		{"mapped ipv4 normalizes", "10.1.2.3:51000", []string{"::ffff:203.0.113.9"}, "203.0.113.9"},
		{"zone is stripped", "10.1.2.3:51000", []string{"fe80::1%eth0"}, "fe80::1"},
		{"malformed hop degrades to proxy", "10.1.2.3:51000", []string{"203.0.113.9, unknown"}, "10.1.2.3"},
		{"blank hop degrades to proxy", "10.1.2.3:51000", []string{"203.0.113.9,, 10.9.9.9"}, "10.1.2.3"},
		{"mapped ipv6 socket peer", "[::ffff:10.1.2.3]:51000", []string{"203.0.113.9"}, "203.0.113.9"},
	}
	for _, tc := range cases {
		got, err := policy.ClientIP(tc.remote, tc.forwarded)
		if err != nil || got != tc.want {
			t.Errorf("%s: ClientIP(%q, %q) = %q, %v; want %q", tc.name, tc.remote, tc.forwarded, got, err, tc.want)
		}
	}
	long := strings.Repeat("203.0.113.9,", maximumForwardedHops) + "203.0.113.9"
	got, err := policy.ClientIP("10.1.2.3:51000", []string{long})
	if err != nil || got != "10.1.2.3" {
		t.Fatalf("oversized chain = %q, %v", got, err)
	}
}
