package claimpass

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var (
	errLost    = errors.New("claim lost")
	errHandle  = errors.New("handler failed")
	errRelease = errors.New("release failed")
)

// recorder keeps the order in which handlers ended and claims were released.
type recorder struct {
	mu     sync.Mutex
	events []string
}

func (r *recorder) record(event string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *recorder) list() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.events)
}

// A cancelled pass joins every handler before it releases that handler's
// claim, and the release still runs with a live, bounded context.
func TestRunJoinsEachHandlerBeforeReleasingItsClaim(t *testing.T) {
	claims := []string{"first", "second"}
	ctx, cancel := context.WithCancel(t.Context())
	var entered sync.WaitGroup
	entered.Add(len(claims))
	proceed := make(chan struct{})
	events := &recorder{}
	var releases sync.Map
	type observed struct {
		err         error
		hasDeadline bool
	}
	done := make(chan error, 1)
	go func() {
		_, err := Run(ctx, claims, Lifecycle[string]{
			Handle: func(handlerContext context.Context, claim string) (bool, error) {
				entered.Done()
				<-handlerContext.Done()
				<-proceed
				events.record("handled " + claim)
				return false, handlerContext.Err()
			},
			Release: func(releaseContext context.Context, claim string) error {
				_, hasDeadline := releaseContext.Deadline()
				releases.Store(claim, observed{err: releaseContext.Err(), hasDeadline: hasDeadline})
				events.record("released " + claim)
				return nil
			},
			ReleaseBudget: time.Minute, Lost: errLost,
		})
		done <- err
	}()
	entered.Wait()
	cancel()
	if got := events.list(); len(got) != 0 {
		t.Fatalf("events before the cancelled handlers returned = %v", got)
	}
	close(proceed)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled pass error = %v, want the handlers' cancellation", err)
	}
	got := events.list()
	for _, claim := range claims {
		handled, released := slices.Index(got, "handled "+claim), slices.Index(got, "released "+claim)
		if handled < 0 || released < handled {
			t.Fatalf("events = %v, want %s handled before it is released", got, claim)
		}
		value, ok := releases.Load(claim)
		if !ok {
			t.Fatalf("claim %s was not released", claim)
		}
		if release := value.(observed); release.err != nil || !release.hasDeadline {
			t.Fatalf("release of %s ran with context error %v and deadline %t, want a live bounded context",
				claim, release.err, release.hasDeadline)
		}
	}
}

// Every claim of a pass is handled at the same time, so one slow subject does
// not hold back another, and the pass runs no more handlers than claims.
func TestRunHandlesTheClaimsOfAPassConcurrently(t *testing.T) {
	claims := []string{"a", "b", "c", "d"}
	// The watchdog turns a sequential implementation into a failure instead
	// of a hang; nothing asserts how long the pass took.
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	allStarted := make(chan struct{})
	var started, running, peak atomic.Int64
	var released atomic.Int64
	_, err := Run(ctx, claims, Lifecycle[string]{
		Handle: func(handlerContext context.Context, _ string) (bool, error) {
			now := running.Add(1)
			defer running.Add(-1)
			for {
				previous := peak.Load()
				if now <= previous || peak.CompareAndSwap(previous, now) {
					break
				}
			}
			if started.Add(1) == int64(len(claims)) {
				close(allStarted)
			}
			select {
			case <-allStarted:
				return true, nil
			case <-handlerContext.Done():
				return false, handlerContext.Err()
			}
		},
		Release: func(context.Context, string) error {
			released.Add(1)
			return nil
		},
		ReleaseBudget: time.Minute, Lost: errLost,
	})
	if err != nil {
		t.Fatalf("pass error = %v, want every handler to meet the others", err)
	}
	if peak.Load() != int64(len(claims)) || released.Load() != int64(len(claims)) {
		t.Fatalf("peak handlers = %d, released = %d, want %d of each", peak.Load(), released.Load(), len(claims))
	}
}

// Handler and release failures are reported per claim in claim order after
// every claim was released; a release that finds its claim lost is not a
// failure.
func TestRunReportsFailuresPerClaimInClaimOrder(t *testing.T) {
	claims := []string{"handler-fails", "release-lost", "release-fails", "both-fail", "clean"}
	var mu sync.Mutex
	released := map[string]int{}
	_, err := Run(t.Context(), claims, Lifecycle[string]{
		Handle: func(_ context.Context, claim string) (bool, error) {
			if claim == "handler-fails" || claim == "both-fail" {
				return false, errors.Join(errHandle, errors.New(claim))
			}
			return false, nil
		},
		Release: func(_ context.Context, claim string) error {
			mu.Lock()
			released[claim]++
			mu.Unlock()
			switch claim {
			case "release-lost":
				return errors.Join(errLost, errors.New(claim))
			case "release-fails", "both-fail":
				return errors.Join(errRelease, errors.New(claim))
			}
			return nil
		},
		ReleaseBudget: time.Minute, Lost: errLost,
	})
	for _, claim := range claims {
		if released[claim] != 1 {
			t.Fatalf("claim %s released %d times, want once", claim, released[claim])
		}
	}
	if errors.Is(err, errLost) {
		t.Fatalf("pass error %v reports a lost claim on release", err)
	}
	joined, ok := err.(interface{ Unwrap() []error })
	if !ok {
		t.Fatalf("pass error = %v, want the joined failures of each claim", err)
	}
	perClaim := joined.Unwrap()
	want := []struct {
		claim             string
		handler, released bool
	}{
		{claim: "handler-fails", handler: true},
		{claim: "release-fails", released: true},
		{claim: "both-fail", handler: true, released: true},
	}
	if len(perClaim) != len(want) {
		t.Fatalf("per-claim failures = %v, want %d in claim order", perClaim, len(want))
	}
	for index, expected := range want {
		failure := perClaim[index]
		if errors.Is(failure, errHandle) != expected.handler || errors.Is(failure, errRelease) != expected.released ||
			!containsMessage(failure, expected.claim) {
			t.Fatalf("failure %d = %v, want %s with handler failure %t and release failure %t",
				index, failure, expected.claim, expected.handler, expected.released)
		}
	}
}

func containsMessage(err error, message string) bool {
	if err == nil {
		return false
	}
	if err.Error() == message {
		return true
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		return slices.ContainsFunc(joined.Unwrap(), func(inner error) bool { return containsMessage(inner, message) })
	}
	return containsMessage(errors.Unwrap(err), message)
}

// A pass reports a change when any handler changed state, and an empty pass
// does nothing.
func TestRunReportsWhetherAnyHandlerChangedState(t *testing.T) {
	lifecycle := Lifecycle[string]{
		Handle:        func(_ context.Context, claim string) (bool, error) { return claim == "changes", nil },
		Release:       func(context.Context, string) error { return nil },
		ReleaseBudget: time.Minute, Lost: errLost,
	}
	for _, test := range []struct {
		claims []string
		want   bool
	}{
		{claims: nil, want: false},
		{claims: []string{"idle"}, want: false},
		{claims: []string{"idle", "changes", "idle"}, want: true},
	} {
		changed, err := Run(t.Context(), test.claims, lifecycle)
		if err != nil || changed != test.want {
			t.Fatalf("pass over %v = (%t, %v), want (%t, nil)", test.claims, changed, err, test.want)
		}
	}
}

// A release that does not finish within its budget ends with the budget and
// is reported; it never runs unbounded.
func TestRunBoundsEachReleaseByItsBudget(t *testing.T) {
	_, err := Run(t.Context(), []string{"stuck"}, Lifecycle[string]{
		Handle: func(context.Context, string) (bool, error) { return false, nil },
		Release: func(releaseContext context.Context, _ string) error {
			<-releaseContext.Done()
			return releaseContext.Err()
		},
		ReleaseBudget: time.Millisecond, Lost: errLost,
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stuck release error = %v, want the end of its budget", err)
	}
}

// An incomplete lifecycle is rejected before any claim is handled.
func TestRunRejectsAnIncompleteLifecycle(t *testing.T) {
	handled := false
	complete := Lifecycle[string]{
		Handle: func(context.Context, string) (bool, error) {
			handled = true
			return false, nil
		},
		Release:       func(context.Context, string) error { return nil },
		ReleaseBudget: time.Minute, Lost: errLost,
	}
	for name, edit := range map[string]func(*Lifecycle[string]){
		"handler":        func(l *Lifecycle[string]) { l.Handle = nil },
		"release":        func(l *Lifecycle[string]) { l.Release = nil },
		"release budget": func(l *Lifecycle[string]) { l.ReleaseBudget = 0 },
		"lost error":     func(l *Lifecycle[string]) { l.Lost = nil },
	} {
		lifecycle := complete
		edit(&lifecycle)
		if _, err := Run(t.Context(), []string{"claim"}, lifecycle); !errors.Is(err, errInvalidLifecycle) {
			t.Fatalf("lifecycle without a %s: error = %v", name, err)
		}
	}
	if handled {
		t.Fatal("an incomplete lifecycle handled a claim")
	}
}

type releaseKey struct{}

// Release on its own: it keeps the caller's
// values but not its cancellation, reports a definite failure, and treats a
// lost claim as released (PER-R26).
func TestReleaseReportsADefiniteFailureAndNotALostClaim(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.WithValue(t.Context(), releaseKey{}, "pass"))
	cancel()
	errStore := errors.New("store refused the release")
	for name, test := range map[string]struct {
		result, want error
	}{
		"released": {},
		"lost":     {result: fmt.Errorf("wrapped: %w", errLost)},
		"refused":  {result: errStore, want: errStore},
	} {
		var live bool
		var value any
		err := Release(cancelled, time.Minute, errLost, func(ctx context.Context) error {
			live, value = ctx.Err() == nil, ctx.Value(releaseKey{})
			return test.result
		})
		if !errors.Is(err, test.want) || (test.want == nil && err != nil) {
			t.Fatalf("%s release: error = %v, want %v", name, err, test.want)
		}
		if !live || value != "pass" {
			t.Fatalf("%s release ran under live=%t value=%v, want a live context with the pass's values", name, live, value)
		}
	}
	for name, call := range map[string]func() error{
		"release":        func() error { return Release(t.Context(), time.Minute, errLost, nil) },
		"release budget": func() error { return Release(t.Context(), 0, errLost, func(context.Context) error { return nil }) },
		"lost error": func() error {
			return Release(t.Context(), time.Minute, nil, func(context.Context) error { return nil })
		},
	} {
		if err := call(); !errors.Is(err, errInvalidLifecycle) {
			t.Fatalf("release without a %s: error = %v", name, err)
		}
	}
}
