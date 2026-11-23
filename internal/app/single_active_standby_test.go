package app

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// Shutdown cancels a standby's context, which also stops its readiness
// listener, so the standby loop can find the listener's end ready together
// with the context's and select either. The listener's end then reports the
// cancellation, which the Server treats as a clean exit (FAIL-PER-2); a
// listener that stops while the context is live still stops the Server with
// its error. The test drives the case the loop's random choice reaches only
// sometimes, so it fails every run if that case loses the cancellation.
func TestControlPlaneLeaseStandbyListenerEndAfterShutdownIsCancellation(t *testing.T) {
	shutDown, cancel := context.WithCancel(t.Context())
	cancel()
	shutdownErr := errors.New("shutdown HTTP server: context deadline exceeded")
	for _, serveErr := range []error{nil, shutdownErr} {
		err := standbyListenerStopped(shutDown, serveErr)
		// runServer exits without an error on exactly this condition.
		if !errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "readiness listener stopped") {
			t.Errorf("listener end %v after shutdown = %v, want the cancellation", serveErr, err)
		}
	}

	live := t.Context()
	err := standbyListenerStopped(live, nil)
	if errors.Is(err, context.Canceled) ||
		err.Error() != "standby readiness listener stopped: listener exited before lease acquisition" {
		t.Errorf("listener exit while live = %v, want listener exited before lease acquisition", err)
	}
	listenerErr := errors.New("serve HTTP: accept tcp: use of closed network connection")
	err = standbyListenerStopped(live, listenerErr)
	if !errors.Is(err, listenerErr) || errors.Is(err, context.Canceled) ||
		!strings.HasPrefix(err.Error(), "standby readiness listener stopped: ") {
		t.Errorf("listener failure while live = %v, want standby readiness listener stopped with %v", err, listenerErr)
	}
}
