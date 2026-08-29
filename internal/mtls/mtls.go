// Package mtls builds role-specific TLS configurations for Contractor's
// private Control Plane and Runtime Agent connections.
package mtls

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"strings"
)

const ControlPlaneURIPrefix = "urn:contractor:control-plane:"

var ErrControlPlaneRole = errors.New("peer certificate is not a Contractor Control Plane certificate")

type Files struct {
	Certificate string
	PrivateKey  string
	CA          string
}

// ControlPlaneServerConfig authenticates every private client against the
// deployment CA. All CA-valid Runtime Agent certificates have equal trust.
func ControlPlaneServerConfig(files Files) (*tls.Config, error) {
	identity, roots, err := load(files)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{identity},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    roots,
	}, nil
}

// RuntimeAgentServerConfig authenticates private callers against the CA and
// additionally requires the reserved Control Plane URI SAN on the peer leaf.
func RuntimeAgentServerConfig(files Files) (*tls.Config, error) {
	result, err := ControlPlaneServerConfig(files)
	if err != nil {
		return nil, err
	}
	result.VerifyConnection = VerifyControlPlanePeer
	return result, nil
}

// ControlPlaneClientConfig authenticates an Agent endpoint using normal
// deployment-CA chain and DNS/IP hostname verification.
func ControlPlaneClientConfig(files Files, serverName string) (*tls.Config, error) {
	return clientConfig(files, serverName, false)
}

// ControlPlaneEndpointClientConfig authenticates Agent endpoints selected at
// runtime. net/http fills ServerName from each request URL before the TLS
// handshake, preserving normal DNS/IP verification across multiple agents.
func ControlPlaneEndpointClientConfig(files Files) (*tls.Config, error) {
	identity, roots, err := load(files)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{identity},
		RootCAs:      roots,
	}, nil
}

// RuntimeAgentClientConfig authenticates the endpoint normally and then
// applies the additional Control Plane URI SAN role check.
func RuntimeAgentClientConfig(files Files, serverName string) (*tls.Config, error) {
	return clientConfig(files, serverName, true)
}

func clientConfig(files Files, serverName string, requireControlPlane bool) (*tls.Config, error) {
	if strings.TrimSpace(serverName) == "" {
		return nil, errors.New("TLS server name is required")
	}
	identity, roots, err := load(files)
	if err != nil {
		return nil, err
	}
	result := &tls.Config{
		MinVersion:   tls.VersionTLS13,
		Certificates: []tls.Certificate{identity},
		RootCAs:      roots,
		ServerName:   serverName,
	}
	if requireControlPlane {
		result.VerifyConnection = VerifyControlPlanePeer
	}
	return result, nil
}

// VerifyControlPlanePeer is used only as tls.Config.VerifyConnection after
// Go's normal chain, validity, EKU, and endpoint-name checks have succeeded.
func VerifyControlPlanePeer(state tls.ConnectionState) error {
	if len(state.VerifiedChains) == 0 || len(state.PeerCertificates) == 0 {
		return fmt.Errorf("%w: peer has no verified certificate chain", ErrControlPlaneRole)
	}
	for _, uri := range state.PeerCertificates[0].URIs {
		value := uri.String()
		if strings.HasPrefix(value, ControlPlaneURIPrefix) && len(value) > len(ControlPlaneURIPrefix) {
			return nil
		}
	}
	return fmt.Errorf("%w: required URI SAN is absent", ErrControlPlaneRole)
}

func load(files Files) (tls.Certificate, *x509.CertPool, error) {
	if strings.TrimSpace(files.Certificate) == "" || strings.TrimSpace(files.PrivateKey) == "" || strings.TrimSpace(files.CA) == "" {
		return tls.Certificate{}, nil, errors.New("certificate, private key, and CA files are required")
	}
	identity, err := tls.LoadX509KeyPair(files.Certificate, files.PrivateKey)
	if err != nil {
		return tls.Certificate{}, nil, fmt.Errorf("load TLS identity: %w", err)
	}
	caPEM, err := os.ReadFile(files.CA)
	if err != nil {
		return tls.Certificate{}, nil, fmt.Errorf("read deployment CA: %w", err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(caPEM) {
		return tls.Certificate{}, nil, errors.New("deployment CA file contains no certificates")
	}
	return identity, roots, nil
}
