package session

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The session service pre-checks the Planner session state against the bound
// registered for the planner_session_state_size CHECK (LIM-PLN-9); the Run
// store test ties that row to the CHECK of the migrated schema.
func TestSessionStateBoundIsItsRegistryRow(t *testing.T) {
	entry := contractstest.ReadLimitsRegistry(t).Naming(t, "planner_session_state_size")
	if registered := entry.Bytes(t); registered != maxSessionJSONBytes {
		t.Fatalf("%s registers %d bytes, but the session service pre-checks %d", entry.ID, registered, maxSessionJSONBytes)
	}
}
