// Package contractstest is the one way Go tests reach the shared fixtures
// under api/testdata that another implementation (the Python Runtime, the UI)
// reads too, so tests at any package depth address them by their repository
// path.
//
// A test that reads a shared fixture is named TestShared*: Path fails any
// other test, so make test-wire-cross-language, which runs every TestShared*
// test, runs every Go reader of every shared fixture. When
// CONTRACTOR_SHARED_FIXTURE_LOG names a file, Path appends each shared
// fixture a test reads to it, which the gate uses to prove that each fixture
// is read on every side its readers manifest declares.
package contractstest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// sharedTestPrefix starts the name of every test that reads a shared fixture.
const sharedTestPrefix = "TestShared"

// sharedFixtureLogEnv names the file that records shared fixture reads.
const sharedFixtureLogEnv = "CONTRACTOR_SHARED_FIXTURE_LOG"

// sharedFixtureRoot is the repository directory holding the shared fixtures.
const sharedFixtureRoot = "api/testdata"

var logMutex sync.Mutex

// Path returns the absolute path of a repository-relative file. The
// repository root is the nearest ancestor of the working directory holding
// go.mod. A path inside api/testdata is a shared fixture: only a TestShared*
// test may address it, and the read is recorded for the shared fixture gate.
func Path(t testing.TB, elem ...string) string {
	t.Helper()
	root := repositoryRoot(t)
	path := filepath.Join(append([]string{root}, elem...)...)
	relative, err := filepath.Rel(root, path)
	if err != nil {
		t.Fatal(err)
	}
	relative = filepath.ToSlash(relative)
	if relative == sharedFixtureRoot || strings.HasPrefix(relative, sharedFixtureRoot+"/") {
		sharedFixture(t, root, relative, path)
	}
	return path
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

func repositoryRoot(t testing.TB) string {
	t.Helper()
	directory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return directory
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			t.Fatal("repository root with go.mod not found")
		}
		directory = parent
	}
}

// sharedFixture holds the test that addresses a shared fixture to the naming
// convention and records a file it reads. Directories, such as the base of a
// glob, are not fixtures and are not recorded.
func sharedFixture(t testing.TB, root, relative, path string) {
	t.Helper()
	test, _, _ := strings.Cut(t.Name(), "/")
	if !strings.HasPrefix(test, sharedTestPrefix) {
		t.Fatalf("%s reads the shared fixture %s; a test that reads api/testdata is named %s* so make test-wire-cross-language runs it",
			test, relative, sharedTestPrefix)
	}
	logPath := os.Getenv(sharedFixtureLogEnv)
	if logPath == "" {
		return
	}
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return
	}
	directory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := filepath.Rel(root, directory)
	if err != nil {
		t.Fatal(err)
	}
	line := fmt.Sprintf("go\t%s\t%s\t%s\n", filepath.ToSlash(pkg), test, relative)
	logMutex.Lock()
	defer logMutex.Unlock()
	file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open the shared fixture log: %v", err)
	}
	// One write per line keeps lines whole when test binaries append at once.
	if _, err := file.WriteString(line); err != nil {
		_ = file.Close()
		t.Fatalf("record the shared fixture read: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("record the shared fixture read: %v", err)
	}
}
