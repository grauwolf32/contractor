package operations

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type testOperation struct {
	body func(context.Context, Call) (stageresult.StageContentResult, error)
}

func (testOperation) Ref() string { return "test@1" }
func (o testOperation) Run(ctx context.Context, call Call) (stageresult.StageContentResult, error) {
	return o.body(ctx, call)
}

func TestRegistryRejectsUndeclaredPublication(t *testing.T) {
	revision := "r"
	registry, err := NewRegistry(testOperation{body: func(context.Context, Call) (stageresult.StageContentResult, error) {
		return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "done", Artifacts: map[string]contracts.ArtifactRef{"output": {Namespace: "other", Name: "file", Revision: &revision}}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = registry.Run(t.Context(), "test@1", Call{RunID: "run", StageExecutionID: "stage", ClaimID: "claim", Outputs: map[string]contracts.ArtifactRef{"output": {Namespace: "expected", Name: "file"}}})
	if err == nil {
		t.Fatal("accepted unrelated output binding")
	}
}

func TestRegistryJoinsCancellationBeforeReturning(t *testing.T) {
	started, joined := make(chan struct{}), make(chan struct{})
	registry, err := NewRegistry(testOperation{body: func(ctx context.Context, _ Call) (stageresult.StageContentResult, error) {
		close(started)
		<-ctx.Done()
		close(joined)
		return stageresult.StageContentResult{}, ctx.Err()
	}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := registry.Run(ctx, "test@1", Call{RunID: "run", StageExecutionID: "stage", ClaimID: "claim"})
		done <- err
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	select {
	case <-joined:
	default:
		t.Fatal("operation outlived return")
	}
}

func TestRegistryClassifiesPinnedDomainConflicts(t *testing.T) {
	for _, wrapped := range []bool{false, true} {
		t.Run(fmt.Sprint(wrapped), func(t *testing.T) {
			original := &Failure{Code: apierror.Conflict.Code(), Message: "The assigned output already exists", Retryable: true}
			var reported error = original
			if wrapped {
				reported = fmt.Errorf("operation: %w", original)
			}
			registry, err := NewRegistry(testOperation{body: func(context.Context, Call) (stageresult.StageContentResult, error) {
				return stageresult.StageContentResult{}, reported
			}})
			if err != nil {
				t.Fatal(err)
			}
			_, err = registry.Run(t.Context(), "test@1", Call{RunID: "run", StageExecutionID: "stage", ClaimID: "claim"})
			var failure *Failure
			if !errors.As(err, &failure) || failure.Code != original.Code || failure.Message != original.Message || failure.Retryable {
				t.Fatalf("pinned conflict classification: %v", err)
			}
			if !original.Retryable {
				t.Fatal("registry mutated the operation's error")
			}
		})
	}
}

func TestRegistryPreservesOtherOperationErrors(t *testing.T) {
	for _, original := range []error{
		&Failure{Code: apierror.StorageTransactionConflict.Code(), Message: "retry transaction", Retryable: true},
		apierror.New(apierror.Conflict, "raw ownership fence"),
		context.Canceled,
	} {
		registry, err := NewRegistry(testOperation{body: func(context.Context, Call) (stageresult.StageContentResult, error) {
			return stageresult.StageContentResult{}, original
		}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = registry.Run(t.Context(), "test@1", Call{RunID: "run", StageExecutionID: "stage", ClaimID: "claim"})
		if err != original {
			t.Fatalf("changed unrelated error: %v", err)
		}
	}
}
