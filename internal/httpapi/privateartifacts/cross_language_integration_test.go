//go:build integration

package privateartifacts

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"net"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestCrossLanguagePrivateArtifactLifecycle(t *testing.T) {
	repositoryRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	temporary := t.TempDir()
	pkiRoot := filepath.Join(temporary, "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{
		DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}
	controlPlane, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(pkiRoot, "artifact-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}

	repository := newMemoryRepository()
	agentPEM, err := os.ReadFile(agent.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(agentPEM)
	if block == nil {
		t.Fatal("Agent certificate is not PEM")
	}
	agentCertificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	principalID, err := mtls.RuntimeAgentID(agentCertificate)
	if err != nil {
		t.Fatal(err)
	}
	grant := testGrant("run-a")
	grant.RuntimeAgentID = principalID
	registry := &fakeRegistry{grant: grant}
	handler, err := NewHandler(Dependencies{
		Registry: registry, Artifacts: artifacts.NewService(repository),
	})
	if err != nil {
		t.Fatal(err)
	}
	tlsConfig, err := mtls.ControlPlaneServerConfig(mtls.Files{
		Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate,
	})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(handler)
	server.TLS = tlsConfig
	server.StartTLS()
	defer server.Close()

	probe := func(arguments ...string) map[string]any {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		baseArguments := []string{
			"tests/artifact_client_probe.py",
			"--api-url", server.URL + "/private/v1",
			"--allocation-id", "allocation-1",
			"--instance-id", "runtime-1",
			"--ca", ca.Certificate,
			"--certificate", agent.Certificate,
			"--private-key", agent.PrivateKey,
		}
		command := exec.CommandContext(
			ctx,
			filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python"),
			append(baseArguments, arguments...)...,
		)
		command.Dir = filepath.Join(repositoryRoot, "runtime")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("Python ArtifactClient probe: %v\n%s", err, output)
		}
		var value map[string]any
		if err := json.Unmarshal(output, &value); err != nil {
			t.Fatalf("decode Python probe output %q: %v", output, err)
		}
		return value
	}

	result := probe()
	created, _ := result["createdRevision"].(string)
	updated, _ := result["updatedRevision"].(string)
	if created == "" || updated == "" || created == updated ||
		result["currentRevision"] != created || result["exactRevision"] != created ||
		result["payload"] != "first payload" || result["outputError"] != "artifact_access_denied" {
		t.Fatalf("unexpected cross-language lifecycle: %+v", result)
	}

	registry.mu.Lock()
	registry.grant.WriteFenced = true
	registry.mu.Unlock()
	fenced := probe("--mode", "fenced", "--expected-revision", updated)
	if fenced["code"] != "allocation_write_fenced" || fenced["retryable"] != false {
		t.Fatalf("fenced cross-language write = %+v", fenced)
	}
	current, exists := repository.current("run-a", "inputs", "source")
	if !exists || current.revision != updated || string(current.data) != "second payload" {
		t.Fatalf("fenced write changed artifact: %+v", current)
	}
}
