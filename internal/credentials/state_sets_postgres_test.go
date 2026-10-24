package credentials

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/statesettest"
)

// TestPostgresCredentialOperationPhasesMatchSchema holds the phase column of
// the migrated credential_operations table to the OperationPhase
// declaration: its CHECK lists exactly the declared constants, and the index
// over unfinished operations admits exactly the prepared ones.
func TestPostgresCredentialOperationPhasesMatchSchema(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	statesettest.Require(t, ctx, pool, statesettest.Table{
		Name: "credential_operations",
		Columns: []statesettest.Column{{
			Name: "phase", Check: "credential_operations_phase_check",
			Values:  statesettest.DeclaredValues(t, "internal/credentials", "OperationPhase"),
			Subsets: map[string][]string{"prepared": {string(OperationPrepared)}},
		}},
		Indexes: map[string]map[string]string{
			"credential_operations_create_identity": nil,
			"credential_operations_prepared_idx":    {"phase": "prepared"},
		},
	})
}

// TestPostgresRuntimeCredentialKindsMatchSchema holds the kind column of the
// migrated runtime_credentials table to runtimesettings.RuntimeCredentialKind,
// the one Server definition the store and the private wire share: its CHECK
// lists exactly the declared kinds.
func TestPostgresRuntimeCredentialKindsMatchSchema(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	statesettest.Require(t, ctx, pool, statesettest.Table{
		Name: "runtime_credentials",
		Columns: []statesettest.Column{{
			Name: "credential_kind", Check: "runtime_credentials_credential_kind_check",
			Values: statesettest.DeclaredValues(t, "internal/contracts/runtimesettings", "RuntimeCredentialKind"),
		}},
		Indexes: map[string]map[string]string{},
	})
}
