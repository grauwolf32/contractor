package contractstest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// namedTB is a test whose name a contractstest test chooses and whose fatal
// failure it observes instead of failing.
type namedTB struct {
	testing.TB
	name    string
	failure string
}

// stopped ends a namedTB's test at its fatal failure, as runtime.Goexit ends
// a real one.
type stopped struct{}

func (n *namedTB) Name() string { return n.name }

func (n *namedTB) Helper() {}

func (n *namedTB) Fatal(args ...any) {
	n.failure = fmt.Sprint(args...)
	panic(stopped{})
}

func (n *namedTB) Fatalf(format string, args ...any) {
	n.failure = fmt.Sprintf(format, args...)
	panic(stopped{})
}

// addressAs resolves a repository path as the named test and returns the
// fatal failure it reported, if any.
func addressAs(name string, elem ...string) (failure string) {
	tb := &namedTB{name: name}
	defer func() {
		if recovered := recover(); recovered != nil {
			if _, ok := recovered.(stopped); !ok {
				panic(recovered)
			}
		}
		failure = tb.failure
	}()
	Path(tb, elem...)
	return ""
}

func TestPathHoldsSharedFixturesToSharedTests(t *testing.T) {
	t.Setenv(sharedFixtureLogEnv, "")
	readers := []string{"api", "testdata", "readers.json"}
	for _, tc := range []struct {
		test  string
		elem  []string
		fails bool
	}{
		{test: "TestOther", elem: readers, fails: true},
		{test: "TestOther/TestSharedCase", elem: readers, fails: true},
		{test: "TestOther", elem: []string{"api", "testdata"}, fails: true},
		{test: "TestOther", elem: []string{"api", "testdata", "v1alpha1", "..", "index.json"}, fails: true},
		{test: "TestSharedOther", elem: readers},
		{test: "TestSharedOther/case", elem: readers},
		{test: "TestOther", elem: []string{"api", "v1alpha1", "common.schema.json"}},
		{test: "TestOther", elem: []string{"api", "testdata-other", "file.json"}},
	} {
		failure := addressAs(tc.test, tc.elem...)
		if tc.fails != (failure != "") {
			t.Errorf("%s addressing %s: failure %q, want failure %t", tc.test, filepath.Join(tc.elem...), failure, tc.fails)
		}
		if tc.fails && !strings.Contains(failure, "is named TestShared*") {
			t.Errorf("%s addressing %s: failure %q does not name the convention", tc.test, filepath.Join(tc.elem...), failure)
		}
	}
}

func TestPathRecordsTheFixtureFilesASharedTestReads(t *testing.T) {
	log := filepath.Join(t.TempDir(), "reads.log")
	t.Setenv(sharedFixtureLogEnv, log)
	if failure := addressAs("TestSharedOther/case", "api", "testdata", "readers.json"); failure != "" {
		t.Fatal(failure)
	}
	// A directory, such as the base of a glob, is not a fixture.
	if failure := addressAs("TestSharedOther", "api", "testdata", "v1alpha1"); failure != "" {
		t.Fatal(failure)
	}
	recorded, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	if want := "go\tinternal/contracts/contractstest\tTestSharedOther\tapi/testdata/readers.json\n"; string(recorded) != want {
		t.Fatalf("recorded %q, want %q", recorded, want)
	}
}
