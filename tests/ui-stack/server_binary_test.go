//go:build e2e

package uistack

import (
	"fmt"
	"os"
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
