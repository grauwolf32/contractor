package controlplane

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/mtls"
)

// RuntimeControlClient implements the Runtime Agent lifecycle gRPC protocol, including
// principal binding and validation of allocation responses and Worker State.
// It owns the reusable connections of every lifecycle and Worker State call
// (CTL-R164): Release closes the released principal's connections and Close
// ends them all when the Server stops.
type RuntimeControlClient struct {
	preparation  AllocationSpecPreparation
	grantReader  AllocationGrantReader
	client       *http.Client
	transports   *mtls.RuntimeAgentTransports
	channels     *mtls.RuntimeChannels
	timeout      time.Duration
	requireHTTPS bool
}

func NewMTLSRuntimeControlClient(files mtls.Files, timeout time.Duration, grants AllocationGrantReader, preparation AllocationSpecPreparation) (*RuntimeControlClient, error) {
	if grants == nil {
		return nil, errors.New("allocation grant reader is required")
	}
	if timeout <= 0 {
		return nil, errors.New("Runtime Agent request timeout must be positive")
	}
	tlsConfig, err := mtls.ControlPlaneEndpointClientConfig(files)
	if err != nil {
		return nil, fmt.Errorf("build Runtime Agent TLS client: %w", err)
	}
	transports, err := mtls.NewRuntimeAgentTransports(tlsConfig, timeout)
	if err != nil {
		return nil, fmt.Errorf("build Runtime Agent connections: %w", err)
	}
	channels, err := mtls.NewRuntimeChannels(tlsConfig)
	if err != nil {
		transports.Close()
		return nil, err
	}
	return &RuntimeControlClient{
		transports: transports, channels: channels, timeout: timeout, requireHTTPS: true,
		grantReader: grants, preparation: preparation,
	}, nil
}

// Close ends every Runtime Agent connection the client owns. The Server calls
// it after the Scheduler stopped; a call started later fails.
func (c *RuntimeControlClient) Close() {
	if c.channels != nil {
		c.channels.Close()
	}
	if c.transports != nil {
		c.transports.Close()
	}
}

var _ RuntimeLifecycle = (*RuntimeControlClient)(nil)
