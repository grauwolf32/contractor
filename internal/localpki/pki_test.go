package localpki

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"net"
	"net/url"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestRenewLeafPreservesKeyIdentitySANsAndUsages(t *testing.T) {
	initial := time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC)
	issuer := Generator{Now: func() time.Time { return initial }}
	renewer := Generator{Now: func() time.Time { return initial.Add(300 * 24 * time.Hour) }}
	root := t.TempDir()
	if _, err := issuer.InitCA(root, false); err != nil {
		t.Fatal(err)
	}
	leaf := LeafOptions{
		DNSNames:    []string{"runtime.example", "localhost"},
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
	}
	for _, test := range []struct {
		name  string
		issue func() (Paths, error)
		renew func() (Paths, error)
	}{
		{name: "Runtime Agent", issue: func() (Paths, error) { return issuer.IssueAgent(root, "worker-1", leaf) },
			renew: func() (Paths, error) { return renewer.RenewAgent(root, "worker-1") }},
		{name: "Control Plane", issue: func() (Paths, error) {
			return issuer.IssueControlPlane(root, ControlPlaneOptions{
				LeafOptions: leaf, URI: "urn:contractor:control-plane:renewal-test",
			})
		}, renew: func() (Paths, error) { return renewer.RenewControlPlane(root) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			paths, err := test.issue()
			if err != nil {
				t.Fatal(err)
			}
			before := readTestCertificate(t, paths.Certificate)
			keyBefore, err := os.ReadFile(paths.PrivateKey)
			if err != nil {
				t.Fatal(err)
			}
			identityBefore, err := mtls.RuntimeAgentID(before)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := test.renew(); err != nil {
				t.Fatal(err)
			}
			after := readTestCertificate(t, paths.Certificate)
			keyAfter, err := os.ReadFile(paths.PrivateKey)
			if err != nil {
				t.Fatal(err)
			}
			identityAfter, err := mtls.RuntimeAgentID(after)
			if err != nil {
				t.Fatal(err)
			}
			if !after.NotAfter.After(before.NotAfter) ||
				!bytes.Equal(before.RawSubjectPublicKeyInfo, after.RawSubjectPublicKeyInfo) ||
				!bytes.Equal(keyBefore, keyAfter) || identityAfter != identityBefore ||
				before.SerialNumber.Cmp(after.SerialNumber) == 0 {
				t.Fatalf("renewal changed identity or failed to extend validity: before=%s after=%s", before.NotAfter, after.NotAfter)
			}
			if !slices.Equal(before.DNSNames, after.DNSNames) ||
				!slices.EqualFunc(before.IPAddresses, after.IPAddresses, net.IP.Equal) ||
				!slices.EqualFunc(before.URIs, after.URIs, func(a, b *url.URL) bool { return a.String() == b.String() }) ||
				before.KeyUsage != after.KeyUsage || !slices.Equal(before.ExtKeyUsage, after.ExtKeyUsage) {
				t.Fatalf("renewal changed SANs or usages: before=%+v after=%+v", before, after)
			}
		})
	}
}

func TestRenewLeafRejectsMismatchedKeyAndInvalidCAWithoutWriting(t *testing.T) {
	now := time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC)
	generator := Generator{Now: func() time.Time { return now }}
	root := t.TempDir()
	if _, err := generator.InitCA(root, false); err != nil {
		t.Fatal(err)
	}
	leaf := LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	first, err := generator.IssueAgent(root, "first", leaf)
	if err != nil {
		t.Fatal(err)
	}
	second, err := generator.IssueAgent(root, "second", leaf)
	if err != nil {
		t.Fatal(err)
	}
	otherKey, err := os.ReadFile(second.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(first.PrivateKey, otherKey, 0o600); err != nil {
		t.Fatal(err)
	}
	assertRenewFailsWithoutWriting(t, first, func() error {
		_, err := generator.RenewAgent(root, "first")
		return err
	})
	ca := CAPaths(root)
	if err := os.WriteFile(ca.Certificate, []byte("invalid CA"), 0o644); err != nil {
		t.Fatal(err)
	}
	assertRenewFailsWithoutWriting(t, second, func() error {
		_, err := generator.RenewAgent(root, "second")
		return err
	})
}

func assertRenewFailsWithoutWriting(t *testing.T, paths Paths, renew func() error) {
	t.Helper()
	certificate, err := os.ReadFile(paths.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	key, err := os.ReadFile(paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if err := renew(); err == nil {
		t.Fatal("invalid renewal succeeded")
	}
	certificateAfter, certificateErr := os.ReadFile(paths.Certificate)
	keyAfter, keyErr := os.ReadFile(paths.PrivateKey)
	if certificateErr != nil || keyErr != nil || !bytes.Equal(certificate, certificateAfter) || !bytes.Equal(key, keyAfter) {
		t.Fatal("failed renewal changed the certificate or key")
	}
}

func readTestCertificate(t *testing.T, path string) *x509.Certificate {
	t.Helper()
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	block, rest := pem.Decode(encoded)
	if block == nil || len(bytes.TrimSpace(rest)) != 0 {
		t.Fatal("invalid certificate PEM")
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return certificate
}
