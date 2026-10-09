// Package contractstest locates the shared wire fixtures that the contracts
// packages and the Python Runtime both read, so tests at any package depth
// address them by their repository path.
package contractstest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Path returns the absolute path of a repository-relative file. The
// repository root is the nearest ancestor of the working directory holding
// go.mod.
func Path(t testing.TB, elem ...string) string {
	t.Helper()
	directory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return filepath.Join(append([]string{directory}, elem...)...)
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			t.Fatal("repository root with go.mod not found")
		}
		directory = parent
	}
}

// ReadFile returns a repository-relative file.
func ReadFile(t testing.TB, elem ...string) []byte {
	t.Helper()
	path := Path(t, elem...)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return data
}

// ReadFixture returns api/testdata/v1alpha1/<kind>/<filename>, where kind is
// "valid" or "invalid".
func ReadFixture(t testing.TB, kind, filename string) []byte {
	t.Helper()
	return ReadFile(t, "api", "testdata", "v1alpha1", kind, filename)
}

// AssertSemanticJSONEqual fails unless both documents decode to equal values.
func AssertSemanticJSONEqual(t testing.TB, left, right []byte) {
	t.Helper()
	var leftValue, rightValue any
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("decode left JSON: %v", err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("decode right JSON: %v", err)
	}
	if !reflect.DeepEqual(leftValue, rightValue) {
		t.Fatalf("semantic JSON differs\nleft:  %s\nright: %s", left, right)
	}
}
