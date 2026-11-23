package apierror

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// specPlatform is docs/spec/platform, which the registry's owner links are
// relative to.
const specPlatform = "../../docs/spec/platform"

var (
	codePattern = regexp.MustCompile(`^[a-z][a-z0-9]*(_[a-z0-9]+)*$`)
	ownerLink   = regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
)

func TestRegistryEntriesAreComplete(t *testing.T) {
	if len(registry) == 0 {
		t.Fatal("the registry is empty")
	}
	for _, row := range registry {
		kind, ok := Lookup(row.code)
		if !ok || kind.entry != row {
			t.Errorf("%s: Lookup does not return its own entry", row.code)
		}
		if !codePattern.MatchString(row.code) || len(row.code) > 128 {
			t.Errorf("%s: a code is lowercase snake case of at most 128 bytes", row.code)
		}
		switch {
		case row.status == 0 && !row.frame:
			t.Errorf("%s: a code without an HTTP status must be a WebSocket error frame code", row.code)
		case row.status != 0 && (row.status < 400 || row.status > 599 || http.StatusText(row.status) == ""):
			t.Errorf("%s: status %d is not an HTTP error status", row.code, row.status)
		}
		// The Error envelope bounds message to 512 bytes, the Eval envelope
		// to 256.
		maximum := 512
		if strings.HasPrefix(row.code, "eval_") {
			maximum = 256
		}
		if row.message == "" || len(row.message) > maximum {
			t.Errorf("%s: message must hold 1 to %d bytes", row.code, maximum)
		}
		if strings.TrimSpace(row.meaning) == "" || strings.ContainsAny(row.meaning+row.owner, "|\n") {
			t.Errorf("%s: meaning and owner must be non-empty single-line table text", row.code)
		}
		links := ownerLink.FindAllStringSubmatch(row.owner, -1)
		if len(links) == 0 {
			t.Errorf("%s: owner %q links no requirement", row.code, row.owner)
		}
		for _, link := range links {
			if _, err := os.Stat(filepath.Join(specPlatform, link[2])); err != nil {
				t.Errorf("%s: owner link %s does not resolve: %v", row.code, link[2], err)
			}
		}
	}
	if _, ok := Lookup("unregistered_code"); ok {
		t.Fatal("Lookup found an unregistered code")
	}
}

func TestEveryEvalAndAuditPackageCodeIsNeverRetryable(t *testing.T) {
	for _, row := range registry {
		permanent := strings.HasPrefix(row.code, "eval_") || strings.HasPrefix(row.code, "audit_") ||
			strings.HasPrefix(row.code, "git_") || strings.HasPrefix(row.code, "archive_")
		if permanent && row.retryable {
			t.Errorf("%s: Eval, Audit, Git and archive codes are answered with retryable false", row.code)
		}
	}
}

type detailedError struct{ message string }

func (e *detailedError) Error() string         { return "internal: " + e.message }
func (e *detailedError) ErrorKind() Kind       { return GitContentUnsupported }
func (e *detailedError) PublicMessage() string { return e.message }

func TestClassifiedErrorsAnswerWithTheFirstKindOfTheirChain(t *testing.T) {
	parent := New(NotFound, "store resource not found")
	narrow := Narrow(parent, RuntimeLabelUnknown, "selected label")
	if !errors.Is(narrow, parent) || narrow.Error() != "store resource not found: selected label" {
		t.Fatalf("narrow sentinel = %q, matches parent %t", narrow, errors.Is(narrow, parent))
	}
	for _, test := range []struct {
		name    string
		err     error
		kind    Kind
		message string
	}{
		{"sentinel", fmt.Errorf("read: %w", parent), NotFound, NotFound.Message()},
		{"narrow before its parent", fmt.Errorf("pin: %w", narrow), RuntimeLabelUnknown, RuntimeLabelUnknown.Message()},
		{"outer kind first", fmt.Errorf("%w: %w", New(Conflict, "outer"), parent), Conflict, Conflict.Message()},
		{"public message", fmt.Errorf("import: %w", &detailedError{message: "a submodule at \"vendor\""}),
			GitContentUnsupported, "a submodule at \"vendor\""},
		{"unclassified wrapper of a sentinel", fmt.Errorf("plain: %w", errors.Join(errors.New("cause"), parent)),
			NotFound, NotFound.Message()},
		{"sentinel message", fmt.Errorf("bind: %w", New(RuntimeConfigInvalid, "internal",
			WithMessage("selected version is unavailable"))), RuntimeConfigInvalid, "selected version is unavailable"},
		{"narrow message", Narrow(parent, RuntimeConfigInvalid, "selected version",
			WithMessage("selected version is unavailable")), RuntimeConfigInvalid, "selected version is unavailable"},
	} {
		t.Run(test.name, func(t *testing.T) {
			classified, ok := Of(test.err)
			if !ok || classified.ErrorKind() != test.kind || MessageOf(classified) != test.message {
				t.Fatalf("Of(%q) = %v %t", test.err, classified, ok)
			}
		})
	}
	if _, ok := Of(errors.New("unclassified")); ok {
		t.Fatal("an unclassified error reported a Kind")
	}
}

func TestSentinelNeedsARegisteredKind(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a sentinel of the zero Kind was created")
		}
	}()
	_ = New(Kind{}, "unclassified")
}
