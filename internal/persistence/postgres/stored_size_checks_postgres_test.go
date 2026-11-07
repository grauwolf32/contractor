package postgres

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// storedSizeDisposition is one line of the stored-size CHECK registry
// (PER-R44): the registry entries the constraint enforces, the entries it
// only backstops, or the reason it is an internal invariant.
type storedSizeDisposition struct {
	enforces  []string
	above     []string
	invariant string
}

var limitToken = regexp.MustCompile(`^(above:)?(LIM-[A-Z]+-[0-9]+)$`)

// Materialization prevents the planner from deparsing another test schema's
// constraints before it applies the namespace filter. Foreign DDL may remove
// the relation a still-visible catalog row names.
const storedSizeCheckNamesSQL = `
WITH owned_checks AS MATERIALIZED (
    SELECT oid, conname FROM pg_constraint
    WHERE contype = 'c' AND connamespace = current_schema()::regnamespace
)
SELECT conname FROM owned_checks
WHERE pg_get_constraintdef(oid) LIKE '%octet_length(%'
ORDER BY conname`

// TestPostgresStoredSizeChecksHaveADisposition reads every CHECK constraint
// of the migrated schema that measures a stored value with octet_length and
// requires exactly one disposition for it in ../stored_size_checks.txt: a
// limits registry entry that names it as an enforcement point, a registry
// entry it backstops, or the reason it is an internal invariant. A registry
// line for a constraint the schema lacks fails too.
func TestPostgresStoredSizeChecksHaveADisposition(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, storedSizeCheckNamesSQL)
	if err != nil {
		t.Fatal(err)
	}
	var schema []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		schema = append(schema, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(schema) == 0 {
		t.Fatal("the migrated schema has no octet_length CHECK constraint")
	}

	dispositions := readStoredSizeChecks(t)
	registry := contractstest.ReadLimitsRegistry(t)
	for _, name := range schema {
		disposition, ok := dispositions[name]
		if !ok {
			t.Errorf("CHECK %s bounds a stored size but has no disposition in stored_size_checks.txt", name)
			continue
		}
		for _, id := range append(slices.Clone(disposition.enforces), disposition.above...) {
			if _, registered := registry[id]; !registered {
				t.Errorf("CHECK %s cites %s, which limits.md does not register", name, id)
			}
		}
		for _, id := range disposition.enforces {
			if !registry.Names(id, name) {
				t.Errorf("CHECK %s enforces %s, but that row does not name it under Enforced by", name, id)
			}
		}
	}
	for name := range dispositions {
		if !slices.Contains(schema, name) {
			t.Errorf("stored_size_checks.txt lists %s, which the migrated schema does not have", name)
		}
	}
	for id := range registry {
		for _, name := range schema {
			if registry.Names(id, name) && !slices.Contains(dispositions[name].enforces, id) {
				t.Errorf("%s names %s as an enforcement point, but stored_size_checks.txt does not say it enforces %s", id, name, id)
			}
		}
	}
}

// readStoredSizeChecks reads the stored-size CHECK registry. Each line is a
// constraint name followed by its disposition: one or more registry entry IDs
// it enforces and entries it backstops ("above:" and the ID), or "invariant:"
// and a reason.
func readStoredSizeChecks(t *testing.T) map[string]storedSizeDisposition {
	t.Helper()
	source, err := os.ReadFile(filepath.Join("..", "stored_size_checks.txt"))
	if err != nil {
		t.Fatal(err)
	}
	dispositions := map[string]storedSizeDisposition{}
	for line := range strings.SplitSeq(string(source), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || strings.HasPrefix(fields[0], "#") {
			continue
		}
		var disposition storedSizeDisposition
		switch {
		case len(fields) > 2 && fields[1] == "invariant:":
			disposition.invariant = strings.Join(fields[2:], " ")
		case len(fields) > 1:
			for _, token := range fields[1:] {
				match := limitToken.FindStringSubmatch(token)
				switch {
				case match == nil:
					t.Fatalf("stored_size_checks.txt line %q: %q is not LIM-<area>-<n> or above:LIM-<area>-<n>", line, token)
				case match[1] == "":
					disposition.enforces = append(disposition.enforces, match[2])
				default:
					disposition.above = append(disposition.above, match[2])
				}
			}
		default:
			t.Fatalf("stored_size_checks.txt line %q has no disposition", line)
		}
		if _, repeated := dispositions[fields[0]]; repeated {
			t.Fatalf("stored_size_checks.txt repeats %s", fields[0])
		}
		dispositions[fields[0]] = disposition
	}
	return dispositions
}
