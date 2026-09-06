package artifacts

import (
	"context"
	"errors"
	"testing"
)

func TestTransferBudgetIsBoundedReentrantAndReleased(t *testing.T) {
	ctx := WithBlobRuntime(context.Background(), NewBlobRuntime(PostgresBlobStore{}, nil))
	var releases []func()
	for i := 0; i < 4; i++ {
		inner, release, err := AcquireTransfer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
		_, nested, err := AcquireTransfer(inner)
		if err != nil {
			t.Fatal(err)
		}
		nested()
	}
	if _, _, err := AcquireTransfer(ctx); !errors.Is(err, ErrTransferCapacity) {
		t.Fatalf("fifth: %v", err)
	}
	releases[0]()
	releases[0]()
	_, release, err := AcquireTransfer(ctx)
	if err != nil {
		t.Fatal(err)
	}
	release()
	for _, release := range releases {
		release()
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := AcquireTransfer(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
