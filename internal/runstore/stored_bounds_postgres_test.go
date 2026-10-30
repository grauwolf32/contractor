package runstore

import (
	"context"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// maxRunEventDataBytes pre-checks two stored bounds, each a CHECK on the
// jsonb text of its column: Run event data (LIM-API-25) and Planner session
// state (LIM-PLN-9). The CHECK of the migrated schema, the Go constant and the
// registry row that names the CHECK must state one value, so none of the
// three can change alone.
func TestPostgresStoredJSONBoundsMatchTheirRegistryRows(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	registry := contractstest.ReadLimitsRegistry(t)
	for _, check := range []struct{ table, constraint, column string }{
		{"workflow_run_events", "workflow_run_events_data_check", "data"},
		{"planner_sessions", "planner_session_state_size", "state"},
	} {
		stored := storedJSONTextBound(t, ctx, pool, check.table, check.constraint, check.column)
		entry := registry.Naming(t, check.constraint)
		if registered := entry.Bytes(t); stored != maxRunEventDataBytes || registered != maxRunEventDataBytes {
			t.Errorf("%s bounds %d stored bytes and %s registers %d, but the Run store pre-checks %d",
				check.constraint, stored, entry.ID, registered, maxRunEventDataBytes)
		}
	}
}

// storedJSONTextBound returns the maximum octet length of column::text that
// the named CHECK constraint of table admits.
func storedJSONTextBound(t *testing.T, ctx context.Context, pool *pgxpool.Pool, table, constraint, column string) int {
	t.Helper()
	var definition string
	if err := pool.QueryRow(ctx, `
SELECT pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = $1::regclass AND conname = $2 AND contype = 'c'`, table, constraint).Scan(&definition); err != nil {
		t.Fatalf("read %s: %v", constraint, err)
	}
	bounds := regexp.MustCompile(`octet_length\(\(`+regexp.QuoteMeta(column)+`\)::text\) <= ([0-9]+)`).FindAllStringSubmatch(definition, -1)
	if len(bounds) != 1 {
		t.Fatalf("%s states %d upper bounds on %s::text, want one: %s", constraint, len(bounds), column, definition)
	}
	bound, err := strconv.Atoi(bounds[0][1])
	if err != nil {
		t.Fatal(err)
	}
	return bound
}
