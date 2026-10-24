//go:build e2e

package e2e

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

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

var containerServerBuild = serverbuildtest.Cache{DisableCGO: true}

func containerServerBinary(t *testing.T, directory string) string {
	t.Helper()
	source, err := containerServerBuild.Binary(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	input, err := os.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	target := filepath.Join(directory, "contractor-server")
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
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
