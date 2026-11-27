package mtls

import (
	"errors"
	"time"
)

// Private connection bounds are pinned by the shared wire fixtures.
const (
	PrivateListenerIdleTimeout     = time.Minute
	privateConnectionIdleReuse     = 30 * time.Second
	privateConnectionReuseLifetime = 5 * time.Minute
	privateConnectionsPerPeer      = 8
	runtimeAgentTransportCapacity  = 256
	privateRPCMetadataBytes        = 16 << 10
)

// ErrRuntimeChannelsClosed rejects calls after their connection owner closed.
var ErrRuntimeChannelsClosed = errors.New("Runtime Agent connections are closed")
