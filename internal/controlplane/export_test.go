package controlplane

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// Register and Heartbeat let tests drive the Registry without terminating
// TLS: they derive a stable principal from the instance ID, where production
// supplies the certificate-derived principal to RegisterAuthenticated and
// HeartbeatAuthenticated.
func (r *InMemoryRegistry) Register(registration control.AgentRegistration) (AgentSnapshot, error) {
	return r.RegisterAuthenticated(inProcessPrincipal(registration.InstanceID), registration)
}

func (r *InMemoryRegistry) Heartbeat(heartbeat control.AgentHeartbeat) (control.HeartbeatResponse, error) {
	return r.HeartbeatAuthenticated(inProcessPrincipal(heartbeat.InstanceID).RuntimeAgentID, heartbeat)
}

// ReserveAll reserves and commits request at once, without the durable
// pinning that PlacementAllocator performs between the two steps.
func (r *InMemoryRegistry) ReserveAll(request ReservationRequest) ([]Reservation, error) {
	return r.reserveAll(request, nil, false)
}

// ErrAgentNotFound is GetAgent's answer for an instance the Registry does
// not hold.
var ErrAgentNotFound = errors.New("Runtime Agent not found")

// GetAgent returns the Registry's snapshot of one registered instance.
func (r *InMemoryRegistry) GetAgent(instanceID string) (AgentSnapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.agents[instanceID]
	if !ok {
		return AgentSnapshot{}, ErrAgentNotFound
	}
	return snapshotAgent(entry), nil
}

func inProcessPrincipal(instanceID string) AuthenticatedPrincipal {
	sum := sha256.Sum256([]byte("in-process-runtime-principal\x00" + instanceID))
	return AuthenticatedPrincipal{
		RuntimeAgentID: hex.EncodeToString(sum[:]), Labels: []string{}, LabelRevision: 1,
	}
}

// NewRuntimeControlClient builds a client on an injected HTTP client for
// focused protocol tests; production uses NewMTLSRuntimeControlClient.
func NewRuntimeControlClient(client *http.Client) (*RuntimeControlClient, error) {
	if client == nil {
		return nil, errors.New("HTTP client is required")
	}
	clone := *client
	clone.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return errors.New("Runtime Agent redirects are not allowed")
	}
	return &RuntimeControlClient{client: &clone}, nil
}
