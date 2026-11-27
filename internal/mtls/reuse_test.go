package mtls

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
	"time"
)

func TestSharedPrivateConnectionReuseLimits(t *testing.T) {
	var limits struct {
		ListenerIdleTimeoutSeconds int `json:"listenerIdleTimeoutSeconds"`
		ClientIdleReuseSeconds     int `json:"clientIdleReuseSeconds"`
		ReuseLifetimeSeconds       int `json:"reuseLifetimeSeconds"`
		ConnectionsPerPeer         int `json:"connectionsPerPeer"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "private-connection-reuse.json"), &limits); err != nil {
		t.Fatal(err)
	}
	if time.Duration(limits.ListenerIdleTimeoutSeconds)*time.Second != PrivateListenerIdleTimeout ||
		time.Duration(limits.ClientIdleReuseSeconds)*time.Second != privateConnectionIdleReuse ||
		time.Duration(limits.ReuseLifetimeSeconds)*time.Second != privateConnectionReuseLifetime ||
		limits.ConnectionsPerPeer != privateConnectionsPerPeer {
		t.Fatalf("Go reuse limits differ from the shared fixture: %+v", limits)
	}
	if privateConnectionIdleReuse >= PrivateListenerIdleTimeout {
		t.Fatal("client idle reuse must end before the peer listener closes an idle connection")
	}
}
