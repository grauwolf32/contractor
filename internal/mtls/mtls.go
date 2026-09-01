// Package mtls builds role-specific TLS configurations for Contractor's
// private Control Plane and Runtime Agent connections.
package mtls

import (
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
)

const ControlPlaneURIPrefix = "urn:contractor:control-plane:"

var (
	ErrControlPlaneRole     = errors.New("peer certificate is not a Contractor Control Plane certificate")
	ErrRuntimeAgentIdentity = errors.New("peer certificate does not match the Runtime Agent principal")
	runtimeAgentIDPattern   = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

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

// RuntimeAgentID derives the stable, non-secret Runtime Agent principal from
// the exact DER SubjectPublicKeyInfo carried by the authenticated leaf. A
// renewed certificate that reuses the key therefore retains its identity.
func RuntimeAgentID(certificate *x509.Certificate) (string, error) {
	if certificate == nil || len(certificate.RawSubjectPublicKeyInfo) == 0 {
		return "", fmt.Errorf("%w: peer leaf has no SubjectPublicKeyInfo", ErrRuntimeAgentIdentity)
	}
	sum := sha256.Sum256(certificate.RawSubjectPublicKeyInfo)
	return hex.EncodeToString(sum[:]), nil
}

// RuntimeAgentIDFromConnection derives a principal only from a normally
// verified TLS connection. Callers must not use an unverified peer chain as
// authentication input.
func RuntimeAgentIDFromConnection(state *tls.ConnectionState) (string, error) {
	if state == nil || len(state.VerifiedChains) == 0 || len(state.PeerCertificates) == 0 {
		return "", fmt.Errorf("%w: peer has no verified certificate chain", ErrRuntimeAgentIdentity)
	}
	if len(state.VerifiedChains[0]) == 0 {
		return "", fmt.Errorf("%w: verified chain has no leaf", ErrRuntimeAgentIdentity)
	}
	return RuntimeAgentID(state.VerifiedChains[0][0])
}

// BindRuntimeAgentPrincipal clones a normal endpoint-verifying client config
// and adds an SPKI equality check. VerifyConnection executes after Go's chain,
// EKU and DNS/IP SAN verification but before net/http writes request bytes.
func BindRuntimeAgentPrincipal(base *tls.Config, expectedRuntimeAgentID string) (*tls.Config, error) {
	if base == nil || !runtimeAgentIDPattern.MatchString(expectedRuntimeAgentID) {
		return nil, fmt.Errorf("%w: expected principal is invalid", ErrRuntimeAgentIdentity)
	}
	result := base.Clone()
	previous := result.VerifyConnection
	result.VerifyConnection = func(state tls.ConnectionState) error {
		if previous != nil {
			if err := previous(state); err != nil {
				return err
			}
		}
		actual, err := RuntimeAgentIDFromConnection(&state)
		if err != nil {
			return err
		}
		if actual != expectedRuntimeAgentID {
			return fmt.Errorf("%w: SPKI fingerprint differs", ErrRuntimeAgentIdentity)
		}
		return nil
	}
	return result, nil
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
