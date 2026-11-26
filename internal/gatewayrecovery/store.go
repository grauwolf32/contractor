package gatewayrecovery

import (
	"context"
	"errors"
	"sync"
	"time"
)

// These bounds refuse new state rather than evicting live probes or receipts.
// Stage cleanup releases Run, binding and receipt capacity; healthy unused
// routes are removed. Blocked routes retain their retry window until restart.
const (
	maximumRoutes         = 10000
	maximumRuns           = 10000
	maximumBindingsPerRun = 1024
	maximumFailures       = 65536
	maximumBindings       = 100000
)

var errCapacity = errors.New("gateway recovery capacity exhausted")

type Service struct {
	mu                                          sync.Mutex
	failureReceipts                             int
	routes                                      map[string]*routeState
	runs                                        map[string]*runState
	bindings                                    map[bindingKey]*runState
	policy                                      Policy
	now                                         func() time.Time
	workerRequestTimeout, plannerRequestTimeout time.Duration
}

type bindingKey struct{ participant, model string }
type allocationRoute struct {
	key     string
	timeout time.Duration
}

type runState struct {
	id, stageID, ownerID string
	active               bool
	routes               map[string]struct{}
	bindings             map[bindingKey]allocationRoute
	waits                map[bindingKey]string
	failures             map[string]struct{}
	changed              chan struct{}
}

type routeState struct {
	members                         map[string]*runState
	blocked                         bool
	waiters                         int
	code                            string
	failures                        int64
	blockedAt, next, automaticUntil time.Time
	probeID, probeRunID             string
	probeUntil                      time.Time
}

func New(policy Policy, workerRequestTimeout, plannerRequestTimeout time.Duration) (*Service, error) {
	if workerRequestTimeout < time.Second || plannerRequestTimeout < time.Second {
		return nil, ErrInvalid
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	return &Service{
		routes: make(map[string]*routeState), runs: make(map[string]*runState),
		bindings: make(map[bindingKey]*runState), policy: policy, now: time.Now,
		workerRequestTimeout: workerRequestTimeout, plannerRequestTimeout: plannerRequestTimeout,
	}, nil
}

func (r *runState) notify()                               { close(r.changed); r.changed = make(chan struct{}) }
func (r *routeState) automaticExpired(now time.Time) bool { return !now.Before(r.automaticUntil) }
func (r *routeState) activeProbe(now time.Time) bool      { return now.Before(r.probeUntil) }
func (r *routeState) release()                            { r.probeID, r.probeRunID, r.probeUntil = "", "", time.Time{} }
func IsUnavailable(err error) bool                        { return errors.Is(err, ErrUnavailable) }

// checkContext is called before a memory mutation. Once applied, an update is
// acknowledged even if cancellation races it; repeating its ID is safe.
func checkContext(ctx context.Context) error { return context.Cause(ctx) }
