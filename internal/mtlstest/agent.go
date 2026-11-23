// Package mtlstest emulates the Python Runtime Agent's mTLS role checks for Go
// boundary tests. Production Runtime Agent policy lives in
// runtime/src/contractor_runtime/mtls.py.
package mtlstest

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"strings"
)

const controlPlaneURIPrefix = "urn:contractor:control-plane:"

var ErrControlPlaneRole = errors.New("peer certificate is not a Contractor Control Plane certificate")

type Files struct {
	Certificate string
	PrivateKey  string
	CA          string
}

func AgentServerConfig(files Files) (*tls.Config, error) {
	identity, roots, err := load(files)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{identity},
		ClientAuth: tls.RequireAndVerifyClientCert, ClientCAs: roots,
		VerifyConnection: verifyControlPlanePeer,
	}, nil
}

func AgentClientConfig(files Files, serverName string) (*tls.Config, error) {
	if strings.TrimSpace(serverName) == "" {
		return nil, errors.New("TLS server name is required")
	}
	identity, roots, err := load(files)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion: tls.VersionTLS13, Certificates: []tls.Certificate{identity},
		RootCAs: roots, ServerName: serverName,
		VerifyConnection: verifyControlPlanePeer,
	}, nil
}

// Go's normal chain, validity, EKU and endpoint-name checks run first.
func verifyControlPlanePeer(state tls.ConnectionState) error {
	if len(state.VerifiedChains) == 0 || len(state.PeerCertificates) == 0 {
		return fmt.Errorf("%w: peer has no verified certificate chain", ErrControlPlaneRole)
	}
	for _, uri := range state.PeerCertificates[0].URIs {
		value := uri.String()
		if strings.HasPrefix(value, controlPlaneURIPrefix) && len(value) > len(controlPlaneURIPrefix) {
			return nil
		}
	}
	return fmt.Errorf("%w: required URI SAN is absent", ErrControlPlaneRole)
}

func load(files Files) (tls.Certificate, *x509.CertPool, error) {
	if files.Certificate == "" || files.PrivateKey == "" || files.CA == "" {
		return tls.Certificate{}, nil, errors.New("certificate, private key and CA files are required")
	}
	identity, err := tls.LoadX509KeyPair(files.Certificate, files.PrivateKey)
	if err != nil {
		return tls.Certificate{}, nil, err
	}
	caPEM, err := os.ReadFile(files.CA)
	if err != nil {
		return tls.Certificate{}, nil, err
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(caPEM) {
		return tls.Certificate{}, nil, errors.New("CA file contains no certificates")
	}
	return identity, roots, nil
}
