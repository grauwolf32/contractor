package projectstore

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/statesettest"
)

// TestPostgresProjectStateSetsMatchSchema holds the lifecycle and deletion
// phase columns of the migrated projects table to the Lifecycle and
// DeletionPhase declarations: each CHECK lists exactly the declared
// constants, each constant passes Valid, and the deletion claim index admits
// exactly the deleting Projects.
func TestPostgresProjectStateSetsMatchSchema(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedProjectPool(t, ctx, databaseURL)

	lifecycles := statesettest.DeclaredValues(t, "internal/projectstore", "Lifecycle")
	phases := statesettest.DeclaredValues(t, "internal/projectstore", "DeletionPhase")
	for _, value := range lifecycles {
		if !Lifecycle(value).Valid() {
			t.Errorf("Lifecycle(%q) is declared but not Valid", value)
		}
	}
	for _, value := range phases {
		if !DeletionPhase(value).Valid() {
			t.Errorf("DeletionPhase(%q) is declared but not Valid", value)
		}
	}
	if Lifecycle("not-a-declared-value").Valid() || DeletionPhase("not-a-declared-value").Valid() {
		t.Error("an undeclared Project lifecycle or deletion phase is Valid")
	}

	statesettest.Require(t, ctx, pool, statesettest.Table{
		Name: "projects",
		Columns: []statesettest.Column{
			{
				Name: "lifecycle_state", Check: "projects_lifecycle_state_check", Values: lifecycles,
				Subsets: map[string][]string{"deleting": {string(LifecycleDeleting)}},
			},
			{Name: "deletion_phase", Check: "projects_deletion_phase_check", Values: phases},
		},
		Indexes: map[string]map[string]string{
			"projects_deletion_claim_idx":         {"lifecycle_state": "deleting"},
			"projects_http_target_credential_idx": nil,
		},
	})
}
