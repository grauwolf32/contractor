// Package localpki generates the deliberately small, deployment-local PKI
// used by Contractor's private control plane. It is not a production
// certificate-management system.
package localpki

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const (
	DefaultRoot            = ".local/pki"
	DefaultControlPlaneURI = "urn:contractor:control-plane:local"
	ControlPlaneURIPrefix  = "urn:contractor:control-plane:"
)

var leafNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

type Generator struct {
	Now    func() time.Time
	Random io.Reader
}

type LeafOptions struct {
	DNSNames    []string
	IPAddresses []net.IP
	Force       bool
}

type ControlPlaneOptions struct {
	LeafOptions
	URI string
}

type Paths struct {
	Certificate string
	PrivateKey  string
}

func CAPaths(root string) Paths {
	return Paths{Certificate: filepath.Join(root, "ca.crt"), PrivateKey: filepath.Join(root, "ca.key")}
}

func ControlPlanePaths(root string) Paths {
	return Paths{Certificate: filepath.Join(root, "control-plane.crt"), PrivateKey: filepath.Join(root, "control-plane.key")}
}

func AgentPaths(root, name string) (Paths, error) {
	if !leafNamePattern.MatchString(name) {
		return Paths{}, fmt.Errorf("agent name must match %s", leafNamePattern)
	}
	return Paths{
		Certificate: filepath.Join(root, "agents", name+".crt"),
		PrivateKey:  filepath.Join(root, "agents", name+".key"),
	}, nil
}

func (g Generator) InitCA(root string, force bool) (Paths, error) {
	if err := validateRoot(root); err != nil {
		return Paths{}, err
	}
	now := g.now().UTC()
	key, err := ecdsa.GenerateKey(elliptic.P256(), g.random())
	if err != nil {
		return Paths{}, fmt.Errorf("generate CA key: %w", err)
	}
	serial, err := randomSerial(g.random())
	if err != nil {
		return Paths{}, fmt.Errorf("generate CA serial: %w", err)
	}
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			Organization: []string{"Contractor Local"},
			CommonName:   "Contractor Local Deployment CA",
		},
		NotBefore:             now.Add(-5 * time.Minute),
		NotAfter:              now.Add(10 * 365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLenZero:        true,
	}
	certificateDER, err := x509.CreateCertificate(g.random(), template, template, &key.PublicKey, key)
	if err != nil {
		return Paths{}, fmt.Errorf("create CA certificate: %w", err)
	}
	paths := CAPaths(root)
	if err := writeKeyPair(paths, certificateDER, key, force); err != nil {
		return Paths{}, err
	}
	return paths, nil
}

func (g Generator) IssueControlPlane(root string, options ControlPlaneOptions) (Paths, error) {
	uri := options.URI
	if uri == "" {
		uri = DefaultControlPlaneURI
	}
	parsedURI, err := url.Parse(uri)
	if err != nil || !strings.HasPrefix(uri, ControlPlaneURIPrefix) || len(uri) == len(ControlPlaneURIPrefix) || parsedURI.String() != uri {
		return Paths{}, fmt.Errorf("Control Plane URI SAN must use %s<non-empty-id>", ControlPlaneURIPrefix)
	}
	return g.issueLeaf(
		root,
		ControlPlanePaths(root),
		"Contractor Control Plane",
		options.LeafOptions,
		[]*url.URL{parsedURI},
	)
}

func (g Generator) IssueAgent(root, name string, options LeafOptions) (Paths, error) {
	paths, err := AgentPaths(root, name)
	if err != nil {
		return Paths{}, err
	}
	return g.issueLeaf(root, paths, "Contractor Runtime Agent "+name, options, nil)
}

func (g Generator) issueLeaf(
	root string,
	paths Paths,
	commonName string,
	options LeafOptions,
	uriSANs []*url.URL,
) (Paths, error) {
	if err := validateRoot(root); err != nil {
		return Paths{}, err
	}
	if err := validateEndpointSANs(options.DNSNames, options.IPAddresses); err != nil {
		return Paths{}, err
	}
	caCertificate, caKey, err := loadCA(root, g.now().UTC())
	if err != nil {
		return Paths{}, err
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), g.random())
	if err != nil {
		return Paths{}, fmt.Errorf("generate leaf key: %w", err)
	}
	serial, err := randomSerial(g.random())
	if err != nil {
		return Paths{}, fmt.Errorf("generate leaf serial: %w", err)
	}
	now := g.now().UTC()
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{Organization: []string{"Contractor Local"}, CommonName: commonName},
		NotBefore:    now.Add(-5 * time.Minute),
		NotAfter:     now.Add(365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		DNSNames:     append([]string(nil), options.DNSNames...),
		URIs:         uriSANs,
	}
	for _, address := range options.IPAddresses {
		template.IPAddresses = append(template.IPAddresses, append(net.IP(nil), address...))
	}
	certificateDER, err := x509.CreateCertificate(g.random(), template, caCertificate, &key.PublicKey, caKey)
	if err != nil {
		return Paths{}, fmt.Errorf("create leaf certificate: %w", err)
	}
	if err := writeKeyPair(paths, certificateDER, key, options.Force); err != nil {
		return Paths{}, err
	}
	return paths, nil
}

func (g Generator) now() time.Time {
	if g.Now != nil {
		return g.Now()
	}
	return time.Now()
}

func (g Generator) random() io.Reader {
	if g.Random != nil {
		return g.Random
	}
	return cryptorand.Reader
}

func loadCA(root string, now time.Time) (*x509.Certificate, *ecdsa.PrivateKey, error) {
	paths := CAPaths(root)
	certificatePEM, err := os.ReadFile(paths.Certificate)
	if err != nil {
		return nil, nil, fmt.Errorf("read CA certificate: %w", err)
	}
	keyPEM, err := os.ReadFile(paths.PrivateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("read CA private key: %w", err)
	}
	certificateBlock, rest := pem.Decode(certificatePEM)
	if certificateBlock == nil || certificateBlock.Type != "CERTIFICATE" || len(bytes.TrimSpace(rest)) != 0 {
		return nil, nil, errors.New("CA certificate file must contain exactly one PEM certificate")
	}
	certificate, err := x509.ParseCertificate(certificateBlock.Bytes)
	if err != nil || !certificate.IsCA || certificate.KeyUsage&x509.KeyUsageCertSign == 0 {
		return nil, nil, errors.New("CA certificate is not a valid certificate authority")
	}
	if now.Before(certificate.NotBefore) || now.After(certificate.NotAfter) {
		return nil, nil, errors.New("CA certificate is outside its validity window")
	}
	keyBlock, rest := pem.Decode(keyPEM)
	if keyBlock == nil || keyBlock.Type != "EC PRIVATE KEY" || len(bytes.TrimSpace(rest)) != 0 {
		return nil, nil, errors.New("CA key file must contain exactly one EC private key")
	}
	key, err := x509.ParseECPrivateKey(keyBlock.Bytes)
	if err != nil || key.Curve != elliptic.P256() {
		return nil, nil, errors.New("CA key is not an ECDSA P-256 private key")
	}
	publicKey, ok := certificate.PublicKey.(*ecdsa.PublicKey)
	if !ok || publicKey.Curve != elliptic.P256() || publicKey.X.Cmp(key.X) != 0 || publicKey.Y.Cmp(key.Y) != 0 {
		return nil, nil, errors.New("CA certificate and private key do not match")
	}
	return certificate, key, nil
}

func validateRoot(root string) error {
	if strings.TrimSpace(root) == "" || strings.ContainsRune(root, 0) {
		return errors.New("PKI root is required")
	}
	return nil
}

func validateEndpointSANs(dnsNames []string, addresses []net.IP) error {
	if len(dnsNames) == 0 && len(addresses) == 0 {
		return errors.New("at least one DNS or IP endpoint SAN is required")
	}
	for _, name := range dnsNames {
		if strings.TrimSpace(name) == "" || strings.ContainsAny(name, "\x00/ ") {
			return fmt.Errorf("invalid DNS SAN %q", name)
		}
	}
	for _, address := range addresses {
		if address == nil {
			return errors.New("invalid IP SAN")
		}
	}
	return nil
}

func randomSerial(random io.Reader) (*big.Int, error) {
	maximum := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 128), big.NewInt(1))
	serial, err := cryptorand.Int(random, maximum)
	if err != nil {
		return nil, err
	}
	return serial.Add(serial, big.NewInt(1)), nil
}

func writeKeyPair(paths Paths, certificateDER []byte, key *ecdsa.PrivateKey, force bool) error {
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return fmt.Errorf("marshal EC private key: %w", err)
	}
	files := []outputFile{
		{path: paths.Certificate, mode: 0o644, data: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificateDER})},
		{path: paths.PrivateKey, mode: 0o600, data: pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})},
	}
	if err := writeFiles(files, force); err != nil {
		return fmt.Errorf("write certificate pair: %w", err)
	}
	return nil
}

type outputFile struct {
	path string
	mode os.FileMode
	data []byte
}

func writeFiles(files []outputFile, force bool) error {
	for _, file := range files {
		if err := os.MkdirAll(filepath.Dir(file.path), 0o700); err != nil {
			return err
		}
		if !force {
			if _, err := os.Lstat(file.path); err == nil {
				return fmt.Errorf("%s already exists (use --force to replace it)", file.path)
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}

	temporary := make([]string, len(files))
	defer func() {
		for _, path := range temporary {
			if path != "" {
				_ = os.Remove(path)
			}
		}
	}()
	for index, file := range files {
		handle, err := os.CreateTemp(filepath.Dir(file.path), ".contractor-pki-*")
		if err != nil {
			return err
		}
		temporary[index] = handle.Name()
		if err := handle.Chmod(file.mode); err != nil {
			_ = handle.Close()
			return err
		}
		if _, err := handle.Write(file.data); err != nil {
			_ = handle.Close()
			return err
		}
		if err := handle.Sync(); err != nil {
			_ = handle.Close()
			return err
		}
		if err := handle.Close(); err != nil {
			return err
		}
	}

	created := make([]string, 0, len(files))
	for index, file := range files {
		if force {
			if err := os.Rename(temporary[index], file.path); err != nil {
				return err
			}
		} else {
			if err := os.Link(temporary[index], file.path); err != nil {
				for _, path := range created {
					_ = os.Remove(path)
				}
				return err
			}
			created = append(created, file.path)
			if err := os.Remove(temporary[index]); err != nil {
				return err
			}
		}
		temporary[index] = ""
	}
	return nil
}
