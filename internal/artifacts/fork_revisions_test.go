package artifacts

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestDescribeForkRevisionsValidatesBeforeStorage(t *testing.T) {
	// A nil repository makes every accidental storage call fail the test.
	service := NewService(nil)
	run, _ := service.Run("original-inputs")
	tooMany := make([]string, MaxExactReadBatchSize+1)
	for i := range tooMany {
		tooMany[i] = fmt.Sprintf("input-%d", i)
	}
	for name, test := range map[string]struct {
		names []string
		want  error
	}{
		"duplicate":  {[]string{"source", "source"}, ErrInvalidName},
		"invalid":    {[]string{"source", "../escape"}, ErrInvalidName},
		"empty name": {[]string{""}, ErrInvalidName},
		"too many":   {tooMany, ErrPayloadTooLarge},
	} {
		t.Run(name, func(t *testing.T) {
			if result, err := run.DescribeForkRevisions(context.Background(), "selected", test.names, LineageContextProjection); result != nil || !errors.Is(err, test.want) {
				t.Fatalf("selection=%v result=%v err=%v, want %v", test.names, result, err, test.want)
			}
		})
	}
	for _, kind := range []ScopeKind{ScopeUser, ScopeProject, ScopeRun} {
		scope, _ := newScope(kind, "original-inputs")
		store, _ := service.Scoped(scope)
		if result, err := store.DescribeForkRevisions(context.Background(), "selected", nil, LineageAuditImport); err != nil || result == nil || len(result) != 0 {
			t.Fatalf("empty selection in scope %s: %v %v", kind, result, err)
		}
	}
	for _, selector := range [][2]string{{"../escape", LineageInputFork}, {"selected", "unknown"}} {
		if _, err := run.DescribeForkRevisions(context.Background(), selector[0], []string{"source"}, selector[1]); !errors.Is(err, ErrInvalidName) {
			t.Fatalf("invalid selector %v: %v", selector, err)
		}
	}
	if _, err := (ScopedStore{service: service}).DescribeForkRevisions(context.Background(), "selected", nil, LineageInputFork); !errors.Is(err, ErrInvalidScope) {
		t.Fatalf("invalid scope: %v", err)
	}
}
