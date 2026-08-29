package main

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/x509"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/localpki"
)

func TestPKICommandsGenerateSecureRoleCertificates(t *testing.T) {
	root := filepath.Join(t.TempDir(), "pki")
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	commands := [][]string{
		{"init-ca", "--root", root},
		{"issue-control-plane", "--root", root},
		{"issue-agent", "--root", root, "--name", "agent-1"},
	}
	for _, command := range commands {
		if err := run(command, &stdout, &stderr); err != nil {
			t.Fatalf("run %v: %v (stderr %s)", command, err, stderr.String())
		}
	}

	ca := localpki.CAPaths(root)
	controlPlane := localpki.ControlPlanePaths(root)
	agent, err := localpki.AgentPaths(root, "agent-1")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{ca.PrivateKey, controlPlane.PrivateKey, agent.PrivateKey} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("private key %s mode = %o, want 600", path, info.Mode().Perm())
		}
	}
	caCertificate := readCertificate(t, ca.Certificate)
	if !caCertificate.IsCA || caCertificate.SignatureAlgorithm != x509.ECDSAWithSHA256 {
		t.Fatalf("CA properties = IsCA %v, signature %v", caCertificate.IsCA, caCertificate.SignatureAlgorithm)
	}
	assertP256(t, caCertificate)
	for _, path := range []string{controlPlane.Certificate, agent.Certificate} {
		certificate := readCertificate(t, path)
		assertP256(t, certificate)
		if certificate.SignatureAlgorithm != x509.ECDSAWithSHA256 || certificate.SerialNumber.Sign() <= 0 ||
			!hasUsage(certificate, x509.ExtKeyUsageClientAuth) || !hasUsage(certificate, x509.ExtKeyUsageServerAuth) {
			t.Fatalf("leaf properties for %s = %+v", path, certificate)
		}
		if err := certificate.CheckSignatureFrom(caCertificate); err != nil {
			t.Fatalf("leaf %s signature: %v", path, err)
		}
	}
	cpCertificate := readCertificate(t, controlPlane.Certificate)
	if len(cpCertificate.URIs) != 1 || cpCertificate.URIs[0].String() != localpki.DefaultControlPlaneURI {
		t.Fatalf("Control Plane URI SANs = %v", cpCertificate.URIs)
	}
	agentCertificate := readCertificate(t, agent.Certificate)
	if len(agentCertificate.URIs) != 0 || agentCertificate.VerifyHostname("localhost") != nil || agentCertificate.VerifyHostname("127.0.0.1") != nil {
		t.Fatalf("Runtime Agent SANs = DNS %v, IP %v, URI %v", agentCertificate.DNSNames, agentCertificate.IPAddresses, agentCertificate.URIs)
	}
}

func TestInitCARefusesOverwriteWithoutForce(t *testing.T) {
	root := filepath.Join(t.TempDir(), "pki")
	if err := run([]string{"init-ca", "--root", root}, &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	paths := localpki.CAPaths(root)
	before, err := os.ReadFile(paths.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"init-ca", "--root", root}, &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
		t.Fatal("second init-ca unexpectedly replaced the CA")
	}
	after, err := os.ReadFile(paths.Certificate)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("CA changed after rejected overwrite: %v", err)
	}
	if err := run([]string{"init-ca", "--root", root, "--force"}, &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("forced init-ca: %v", err)
	}
	replaced, err := os.ReadFile(paths.Certificate)
	if err != nil || bytes.Equal(before, replaced) {
		t.Fatalf("forced CA replacement did not occur: %v", err)
	}
}

func readCertificate(t *testing.T, path string) *x509.Certificate {
	t.Helper()
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(encoded)
	if block == nil {
		t.Fatalf("no PEM certificate in %s", path)
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return certificate
}

func assertP256(t *testing.T, certificate *x509.Certificate) {
	t.Helper()
	key, ok := certificate.PublicKey.(*ecdsa.PublicKey)
	if !ok || key.Curve != elliptic.P256() {
		t.Fatalf("certificate public key is not ECDSA P-256: %T", certificate.PublicKey)
	}
}

func hasUsage(certificate *x509.Certificate, expected x509.ExtKeyUsage) bool {
	for _, usage := range certificate.ExtKeyUsage {
		if usage == expected {
			return true
		}
	}
	return false
}
