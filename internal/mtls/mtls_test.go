package mtls

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

func TestRoleSpecificTLSHandshakes(t *testing.T) {
	files := generateFiles(t, "deployment")

	controlPlaneServer, err := ControlPlaneServerConfig(files.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	runtimeClient, err := RuntimeAgentClientConfig(files.agent, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	if clientErr, serverErr := handshake(runtimeClient, controlPlaneServer); clientErr != nil || serverErr != nil {
		t.Fatalf("Agent -> Control Plane handshake = client %v, server %v", clientErr, serverErr)
	}

	runtimeServer, err := RuntimeAgentServerConfig(files.agent)
	if err != nil {
		t.Fatal(err)
	}
	controlPlaneClient, err := ControlPlaneClientConfig(files.controlPlane, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	if clientErr, serverErr := handshake(controlPlaneClient, runtimeServer); clientErr != nil || serverErr != nil {
		t.Fatalf("Control Plane -> Agent handshake = client %v, server %v", clientErr, serverErr)
	}
}

func TestRuntimeAgentRejectsCAValidPeerWithoutControlPlaneURI(t *testing.T) {
	files := generateFiles(t, "deployment")
	server, err := ControlPlaneServerConfig(files.agent)
	if err != nil {
		t.Fatal(err)
	}
	client, err := RuntimeAgentClientConfig(files.agent, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	clientErr, _ := handshake(client, server)
	if !errors.Is(clientErr, ErrControlPlaneRole) {
		t.Fatalf("client handshake error = %v, want Control Plane role error", clientErr)
	}
}

func TestControlPlaneRejectsForeignRuntimeAgent(t *testing.T) {
	trusted := generateFiles(t, "trusted")
	foreign := generateFiles(t, "foreign")
	server, err := ControlPlaneServerConfig(trusted.controlPlane)
	if err != nil {
		t.Fatal(err)
	}
	foreignIdentityTrustingServer := foreign.agent
	foreignIdentityTrustingServer.CA = trusted.controlPlane.CA
	client, err := ControlPlaneClientConfig(foreignIdentityTrustingServer, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	clientErr, serverErr := handshake(client, server)
	if clientErr == nil && serverErr == nil {
		t.Fatal("foreign Runtime Agent certificate was accepted")
	}
}

func TestNormalVerificationRejectsValidityAndEKUErrors(t *testing.T) {
	files := generateFiles(t, "deployment")
	client, err := ControlPlaneClientConfig(files.controlPlane, "localhost")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		notBefore time.Time
		notAfter  time.Time
		usages    []x509.ExtKeyUsage
	}{
		{name: "expired", notBefore: time.Now().Add(-2 * time.Hour), notAfter: time.Now().Add(-time.Hour), usages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}},
		{name: "not-yet-valid", notBefore: time.Now().Add(time.Hour), notAfter: time.Now().Add(2 * time.Hour), usages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}},
		{name: "missing-server-EKU", notBefore: time.Now().Add(-time.Hour), notAfter: time.Now().Add(time.Hour), usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			invalid := issueCustomCertificate(t, files.root, test.notBefore, test.notAfter, test.usages)
			serverFiles := Files{Certificate: invalid.Certificate, PrivateKey: invalid.PrivateKey, CA: files.controlPlane.CA}
			server, err := ControlPlaneServerConfig(serverFiles)
			if err != nil {
				t.Fatal(err)
			}
			clientErr, _ := handshake(client, server)
			if clientErr == nil {
				t.Fatal("invalid server certificate was accepted")
			}
		})
	}
}

type generatedFiles struct {
	root         string
	controlPlane Files
	agent        Files
}

func generateFiles(t *testing.T, directory string) generatedFiles {
	t.Helper()
	root := filepath.Join(t.TempDir(), directory)
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	options := localpki.LeafOptions{DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	cp, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: options})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent-1", options)
	if err != nil {
		t.Fatal(err)
	}
	return generatedFiles{
		root:         root,
		controlPlane: Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate},
		agent:        Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate},
	}
}

func handshake(clientConfig, serverConfig *tls.Config) (error, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err, err
	}
	defer listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	serverResult := make(chan error, 1)
	go func() {
		raw, acceptErr := listener.Accept()
		if acceptErr != nil {
			serverResult <- acceptErr
			return
		}
		defer raw.Close()
		serverResult <- tls.Server(raw, serverConfig).HandshakeContext(ctx)
	}()
	rawClient, err := net.DialTimeout("tcp", listener.Addr().String(), time.Second)
	if err != nil {
		return err, <-serverResult
	}
	client := tls.Client(rawClient, clientConfig)
	clientErr := client.HandshakeContext(ctx)
	serverErr := <-serverResult
	_ = rawClient.Close()
	return clientErr, serverErr
}

func issueCustomCertificate(
	t *testing.T,
	root string,
	notBefore time.Time,
	notAfter time.Time,
	usages []x509.ExtKeyUsage,
) localpki.Paths {
	t.Helper()
	caCertificate := parseCertificate(t, localpki.CAPaths(root).Certificate)
	caKey := parseKey(t, localpki.CAPaths(root).PrivateKey)
	key, err := ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject:      pkix.Name{CommonName: "invalid test peer"},
		NotBefore:    notBefore,
		NotAfter:     notAfter,
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  usages,
		DNSNames:     []string{"localhost"},
	}
	der, err := x509.CreateCertificate(cryptorand.Reader, template, caCertificate, &key.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	paths := localpki.Paths{Certificate: filepath.Join(directory, "peer.crt"), PrivateKey: filepath.Join(directory, "peer.key")}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.Certificate, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.PrivateKey, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		t.Fatal(err)
	}
	return paths
}

func parseCertificate(t *testing.T, path string) *x509.Certificate {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(data)
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return certificate
}

func parseKey(t *testing.T, path string) *ecdsa.PrivateKey {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(data)
	key, err := x509.ParseECPrivateKey(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return key
}
