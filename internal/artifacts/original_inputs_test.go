package artifacts

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestDescribeOriginalInputsValidatesBeforeStorage(t *testing.T) {
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
			if result, err := run.DescribeOriginalInputs(context.Background(), test.names); result != nil || !errors.Is(err, test.want) {
				t.Fatalf("selection=%v result=%v err=%v, want %v", test.names, result, err, test.want)
			}
		})
	}
	for _, kind := range []ScopeKind{ScopeUser, ScopeProject} {
		scope, _ := newScope(kind, "original-inputs")
		store, _ := service.Scoped(scope)
		if _, err := store.DescribeOriginalInputs(context.Background(), []string{"source"}); !errors.Is(err, ErrInvalidScope) {
			t.Fatalf("scope %s: %v", kind, err)
		}
	}
	if result, err := run.DescribeOriginalInputs(context.Background(), nil); err != nil || result == nil || len(result) != 0 {
		t.Fatalf("empty selection=%v err=%v", result, err)
	}
}
