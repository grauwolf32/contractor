//go:build e2e

package e2e

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/tests/internal/serverbuildtest"
)

var processServerBuild serverbuildtest.Cache

func TestMain(m *testing.M) {
	code := m.Run()
	if err := processServerBuild.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		code = 1
	}

	if err := containerServerBuild.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		code = 1
	}
	if err := containerStatusBuild.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		code = 1
	}

	os.Exit(code)
}

func sharedServerBinary(t *testing.T) string {
	t.Helper()
	binary, err := processServerBuild.Binary(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return binary
}

var containerServerBuild serverbuildtest.Cache
var containerStatusBuild serverbuildtest.Cache
var containerTargetOnce sync.Once
var containerTargetError error

func prepareContainerBuild(t *testing.T) {
	t.Helper()
	containerTargetOnce.Do(func() {
		ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
		defer cancel()
		data, err := exec.CommandContext(ctx, "podman", "info", "--format", "{{.Host.Arch}}").Output()
		if err != nil {
			containerTargetError = fmt.Errorf("resolve Podman build target: %w", err)
			return
		}
		arch := strings.TrimSpace(string(data))
		if arch == "" || len(arch) > 16 || strings.ContainsAny(arch, " \t\r\n=") {
			containerTargetError = fmt.Errorf("invalid Podman build architecture")
			return
		}
		containerServerBuild.DisableCGO = true
		containerServerBuild.TargetOS = "linux"
		containerServerBuild.TargetArch = arch
		containerStatusBuild.DisableCGO = true
		containerStatusBuild.TargetOS = "linux"
		containerStatusBuild.TargetArch = arch
		containerStatusBuild.Package = "./tests/e2e/testdata/container-status"
	})
	if containerTargetError != nil {
		t.Fatal(containerTargetError)
	}
}

func containerServerBinary(t *testing.T, directory string) string {
	t.Helper()
	prepareContainerBuild(t)
	source, err := containerServerBuild.Binary(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return copyContainerExecutable(t, source, directory, "contractor-server")
}

func containerStatusBinary(t *testing.T, directory string) {
	t.Helper()
	prepareContainerBuild(t)
	source, err := containerStatusBuild.Binary(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	copyContainerExecutable(t, source, directory, "process-status")
}

func copyContainerExecutable(t *testing.T, source, directory, name string) string {
	t.Helper()
	input, err := os.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	target := filepath.Join(directory, name)
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0755)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(output, input)
	closeErr := output.Close()
	if copyErr != nil {
		t.Fatal(copyErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	return target
}
