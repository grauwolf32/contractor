//go:build integration

package controlplane

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
)

func TestCrossLanguageMTLSAllocationLifecycle(t *testing.T) {
	repositoryRoot, err := filepath.Abs("../..")
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
	agent, err := generator.IssueAgent(pkiRoot, "integration-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	port := unusedIntegrationPort(t)
	workRoot := filepath.Join(temporary, "work")
	processContext, stopProcess := context.WithCancel(context.Background())
	defer stopProcess()
	var processOutput bytes.Buffer
	command := exec.CommandContext(
		processContext,
		filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python"),
		"tests/lifecycle_server.py",
		"--port", fmt.Sprint(port),
		"--ca", ca.Certificate,
		"--certificate", agent.Certificate,
		"--private-key", agent.PrivateKey,
		"--work-root", workRoot,
	)
	command.Dir = filepath.Join(repositoryRoot, "runtime")
	command.Stdout = &processOutput
	command.Stderr = &processOutput
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	processDone := make(chan error, 1)
	go func() { processDone <- command.Wait() }()
	processWaited := false
	waitForProcess := func() error {
		if processWaited {
			return nil
		}
		processWaited = true
		return <-processDone
	}
	defer func() {
		if !processWaited {
			_ = command.Process.Signal(os.Interrupt)
			select {
			case <-processDone:
				processWaited = true
			case <-time.After(3 * time.Second):
				stopProcess()
				_ = waitForProcess()
			}
		}
	}()

	client, err := NewMTLSRuntimeControlClient(mtls.Files{
		Certificate: controlPlane.Certificate, PrivateKey: controlPlane.PrivateKey, CA: ca.Certificate,
	}, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	baseURL := fmt.Sprintf("https://127.0.0.1:%d", port)
	lease := time.Now().Add(time.Minute).UTC()
	reservation := testReservation(
		"allocation_integration", "builder", baseURL, baseURL, testTemplate(t), lease,
	)
	settings := testRuntimeSettings()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var handle contracts.WorkerHandle
	var prepareErr error
	for ctx.Err() == nil {
		handle, prepareErr = client.Prepare(ctx, reservation, settings)
		if prepareErr == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if prepareErr != nil {
		stopProcess()
		_ = waitForProcess()
		t.Fatalf("cross-language prepare failed: %v\nRuntime output:\n%s", prepareErr, processOutput.String())
	}
	if handle.AllocationID != reservation.Grant.AllocationID || handle.AgentTemplateRef != reservation.AgentTemplate.Ref {
		t.Fatalf("unexpected cross-language WorkerHandle: %+v", handle)
	}
	report, err := client.Finalize(ctx, reservation, "finalization_integration", time.Now().Add(10*time.Second))
	if err != nil || !report.Complete {
		t.Fatalf("cross-language finalize = (%+v, %v)", report, err)
	}
	if err := client.Release(ctx, reservation); err != nil {
		t.Fatalf("cross-language release: %v", err)
	}

	if err := command.Process.Signal(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	if err := waitForProcess(); err != nil {
		t.Fatalf("Runtime test server exit: %v\n%s", err, processOutput.String())
	}
}

func unusedIntegrationPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}
