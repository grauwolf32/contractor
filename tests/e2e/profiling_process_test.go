package e2e

import (
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	runtimeprofiling "github.com/grauwolf32/contractor/internal/profiling"
)

const profilingHelperEnvironment = "CONTRACTOR_PROFILING_HELPER"

func TestProfilingSeparateProcessUsesOnlyTheDiagnosticListener(t *testing.T) {
	readyPath := filepath.Join(t.TempDir(), "ready")
	secret := "profiling-process-secret-canary"
	command := exec.Command(os.Args[0], "-test.run=^TestProfilingHelperProcess$")
	command.Env = append(os.Environ(),
		profilingHelperEnvironment+"=1",
		"CONTRACTOR_PROFILING_READY="+readyPath,
		"CONTRACTOR_PROFILING_SECRET="+secret,
	)
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	stopped := false
	defer func() {
		if !stopped {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	}()

	address := waitForProfilingAddress(t, readyPath, command, &output)
	client := &http.Client{Timeout: 3 * time.Second}
	index := profilingProcessGET(t, client, "http://"+address+"/debug/pprof/", http.StatusOK)
	if strings.Contains(string(index), "cmdline") || strings.Contains(string(index), secret) {
		t.Fatalf("diagnostic index exposed a forbidden surface: %q", index)
	}
	profilingProcessGET(t, client, "http://"+address+"/debug/pprof/cmdline", http.StatusNotFound)
	heap := profilingProcessGET(t, client, "http://"+address+"/debug/pprof/heap", http.StatusOK)
	if len(heap) == 0 {
		t.Fatal("heap profile is empty")
	}
	profilingProcessGET(t, client, "http://"+address+"/debug/pprof/profile?seconds=0", http.StatusBadRequest)

	if err := command.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	select {
	case err := <-done:
		stopped = true
		if err != nil {
			t.Fatalf("profiling helper stopped with error: %v: %s", err, output.String())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("profiling helper did not stop within its bound")
	}
	if strings.Contains(output.String(), secret) {
		t.Fatal("profiling helper logged environment content")
	}
}

func TestProfilingHelperProcess(t *testing.T) {
	if os.Getenv(profilingHelperEnvironment) != "1" {
		return
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server, err := runtimeprofiling.NewWithListener(listener, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	readyPath := os.Getenv("CONTRACTOR_PROFILING_READY")
	if readyPath == "" {
		t.Fatal("profiling helper ready path is missing")
	}
	if err := os.WriteFile(readyPath, []byte(server.Addr().String()), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := server.Run(ctx); err != nil {
		t.Fatal(err)
	}
}

func waitForProfilingAddress(
	t *testing.T,
	path string,
	command *exec.Cmd,
	output *bytes.Buffer,
) string {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(path)
		if err == nil && len(data) != 0 {
			return string(data)
		}
		if command.ProcessState != nil {
			t.Fatalf("profiling helper exited before readiness: %s", output.String())
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("profiling helper did not become ready: %s", output.String())
	return ""
}

func profilingProcessGET(
	t *testing.T,
	client *http.Client,
	endpoint string,
	wantStatus int,
) []byte {
	t.Helper()
	response, err := client.Get(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != wantStatus {
		t.Fatalf("GET %s status = %d, want %d: %s", endpoint, response.StatusCode, wantStatus, body)
	}
	return body
}
