// Package mtls builds role-specific TLS configurations for Contractor's
// private Control Plane and Runtime Agent connections.
package mtls

import (
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var ErrRuntimeAgentIdentity = errors.New("peer certificate does not match the Runtime Agent principal")

type Files struct {
	Certificate string
	PrivateKey  string
	CA          string
}

// LeafExpiry reports when the configured leaf expires and whether it falls
// within the caller's warning window. The clock is supplied by the caller so
// startup diagnostics can be tested without waiting for real time to pass.
func LeafExpiry(certificatePath string, now time.Time, warningWindow time.Duration) (time.Time, bool, error) {
	return certificateExpiry("TLS leaf certificate", certificatePath, now, warningWindow)
}

// CAExpiry reports when the deployment CA expires and whether it falls within
// the caller's warning window. A leaf never outlives its CA, so an approaching
// CA expiry breaks every private mTLS link and warrants its own warning.
func CAExpiry(caPath string, now time.Time, warningWindow time.Duration) (time.Time, bool, error) {
	return certificateExpiry("deployment CA certificate", caPath, now, warningWindow)
}

func certificateExpiry(label, path string, now time.Time, warningWindow time.Duration) (time.Time, bool, error) {
	if warningWindow <= 0 {
		return time.Time{}, false, errors.New("certificate expiry warning window must be positive")
	}
	encoded, err := os.ReadFile(path)
	if err != nil {
		return time.Time{}, false, fmt.Errorf("read %s: %w", label, err)
	}
	block, _ := pem.Decode(encoded)
	if block == nil || block.Type != "CERTIFICATE" {
		return time.Time{}, false, fmt.Errorf("%s is not PEM encoded", label)
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return time.Time{}, false, fmt.Errorf("parse %s: %w", label, err)
	}
	return certificate.NotAfter, !now.Add(warningWindow).Before(certificate.NotAfter), nil
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

// HasVerifiedRuntimeAgent reports whether the request's connection presented a
// verified Runtime Agent certificate. Private boundaries use it to honor
// peer-supplied request metadata only from an authenticated Runtime Agent.
func HasVerifiedRuntimeAgent(r *http.Request) bool {
	_, err := RuntimeAgentIDFromConnection(r.TLS)
	return err == nil
}

// BindRuntimeAgentPrincipal clones a normal endpoint-verifying client config
// and adds an SPKI equality check. VerifyConnection executes after Go's chain,
// EKU and DNS/IP SAN verification but before net/http writes request bytes.
func BindRuntimeAgentPrincipal(base *tls.Config, expectedRuntimeAgentID string) (*tls.Config, error) {
	if base == nil || !contracts.ValidRuntimeAgentID(expectedRuntimeAgentID) {
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
