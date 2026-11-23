package gitimport

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
)

// An import that ended with its context is a timeout or an interruption,
// unless one of the import's own errors already explains the failure
// (GIT-R72, GIT-R73).
func TestImportEndedClassifiesContextEndings(t *testing.T) {
	unsupported := &UnsupportedEntryError{Kind: "submodule", Path: "vendor"}
	for _, test := range []struct {
		name string
		err  error
		want apierror.Kind
	}{
		{"deadline", fmt.Errorf("fetch: %w", context.DeadlineExceeded), apierror.GitImportTimeout},
		{"cancelled", fmt.Errorf("fetch: %w", context.Canceled), apierror.GitImportCancelled},
		{"own error first", fmt.Errorf("%w: %w", ErrRemote, context.DeadlineExceeded), apierror.GitRemoteUnavailable},
		{"budget during a deadline", fmt.Errorf("%w: %w", ErrBudget, context.Canceled), apierror.GitImportLimit},
		{"unsupported entry", fmt.Errorf("snapshot: %w", unsupported), apierror.GitContentUnsupported},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := importEnded(test.err)
			classified, ok := apierror.Of(err)
			if !ok || classified.ErrorKind() != test.want || !errors.Is(err, test.err) {
				t.Fatalf("importEnded(%v) = %v", test.err, err)
			}
		})
	}
	if err := importEnded(nil); err != nil {
		t.Fatalf("importEnded(nil) = %v", err)
	}
	unclassified := errors.New("store failure")
	if err := importEnded(unclassified); err != unclassified {
		t.Fatalf("importEnded changed an unrelated error: %v", err)
	}
}
