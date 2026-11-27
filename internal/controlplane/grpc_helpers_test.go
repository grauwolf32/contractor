package controlplane

import (
	"context"
	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"net"
	"path/filepath"
	"testing"
	"time"
)

type testPrincipalRegistrar struct{}

type failingPrincipalRegistrar struct{ err error }

func (f failingPrincipalRegistrar) Register(context.Context, string) (agentprincipal.Principal, error) {
	return agentprincipal.Principal{}, f.err
}

func (testPrincipalRegistrar) Register(
	_ context.Context, runtimeAgentID string,
) (agentprincipal.Principal, error) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	return agentprincipal.Principal{
		RuntimeAgentID: runtimeAgentID, Labels: []string{},
		LabelRevision: 1, CreatedBy: "test", CreatedAt: now, UpdatedBy: "test", UpdatedAt: now,
	}, nil
}

type pkiFiles struct {
	controlPlane mtls.Files
	agent        mtls.Files
}

func generatePKI(t *testing.T, name string) pkiFiles {
	t.Helper()
	root := filepath.Join(t.TempDir(), name)
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	options := localpki.LeafOptions{
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}
	controlPlane, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: options})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "agent-1", options)
	if err != nil {
		t.Fatal(err)
	}
	return pkiFiles{
		controlPlane: mtls.Files{Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate},
		agent:        mtls.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate},
	}
}
