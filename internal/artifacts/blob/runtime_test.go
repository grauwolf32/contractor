package blob_test

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
)

func TestTransferBudgetIsBoundedReentrantAndReleased(t *testing.T) {
	runtime := blob.NewRuntime(inline.Store{}, nil)
	ctx := context.Background()
	var releases []func()
	for i := 0; i < 4; i++ {
		inner, release, err := runtime.AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
		_, nested, err := runtime.AcquireTransfer(inner)
		if err != nil {
			t.Fatal(err)
		}
		nested()
	}
	if _, _, err := runtime.AcquireTransfer(ctx); !errors.Is(err, blob.ErrTransferCapacity) {
		t.Fatalf("fifth: %v", err)
	}
	releases[0]()
	releases[0]()
	_, release, err := runtime.AcquireTransfer(ctx)
	if err != nil {
		t.Fatal(err)
	}
	release()
	for _, release := range releases {
		release()
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := runtime.AcquireTransfer(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
