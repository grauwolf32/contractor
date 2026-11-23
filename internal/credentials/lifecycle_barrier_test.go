package credentials

import (
	"context"
	"testing"
	"time"
)

func TestLifecycleBarrierFencesMutationUntilReferenceCommitCompletes(t *testing.T) {
	t.Parallel()
	barrier := NewLifecycleBarrier()
	referenceEntered := make(chan struct{})
	releaseReference := make(chan struct{})
	referenceDone := make(chan error, 1)
	go func() {
		referenceDone <- barrier.WithCredentialReferences(context.Background(), func() error {
			close(referenceEntered)
			<-releaseReference
			return nil
		})
	}()
	<-referenceEntered

	mutationStarted := make(chan struct{})
	mutationEntered := make(chan struct{})
	mutationDone := make(chan error, 1)
	go func() {
		close(mutationStarted)
		mutationDone <- barrier.WithCredentialMutation(context.Background(), func() error {
			close(mutationEntered)
			return nil
		})
	}()
	<-mutationStarted
	select {
	case <-mutationEntered:
		t.Fatal("credential mutation crossed a live reference fence")
	case <-time.After(25 * time.Millisecond):
	}
	close(releaseReference)
	if err := <-referenceDone; err != nil {
		t.Fatal(err)
	}
	select {
	case <-mutationEntered:
	case <-time.After(time.Second):
		t.Fatal("credential mutation did not resume after reference commit")
	}
	if err := <-mutationDone; err != nil {
		t.Fatal(err)
	}
}
