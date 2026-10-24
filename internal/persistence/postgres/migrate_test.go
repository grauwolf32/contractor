package postgres

import (
	"context"
	"crypto/sha256"
	"errors"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

func TestEmbeddedMigrationsAreOrderedAndExcludeRuntimeLiveness(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(items) == 0 {
		t.Fatal("no migrations loaded")
	}
	previous := int64(0)
	for _, item := range items {
		if item.version <= previous {
			t.Fatalf("migration versions are not increasing: %d after %d", item.version, previous)
		}
		previous = item.version
		lower := strings.ToLower(string(item.contents))
		for _, forbidden := range []string{"runtime_agents", "heartbeat_lease", "create type"} {
			if strings.Contains(lower, forbidden) {
				t.Fatalf("migration %s contains forbidden durable construct %q", item.name, forbidden)
			}
		}
	}
}

// Every pending migration runs inside the migrator's one transaction under
// the budget the operator chose (PER-R31, PER-R32): no migration controls the
// transaction, runs a statement PostgreSQL refuses inside a transaction
// block or names a timeout setting.
func TestEmbeddedMigrationsStayInsideTheMigrationTransaction(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	for _, item := range items {
		if problem := migrationTransactionProblem(string(item.contents)); problem != "" {
			t.Errorf("migration %s %s", item.name, problem)
		}
	}
	for sql, refused := range map[string]bool{
		"COMMIT;":           true,
		"SELECT 1;\nbegin;": true,
		"SAVEPOINT before;": true,
		"CREATE UNIQUE INDEX CONCURRENTLY i ON t (a);": true,
		"VACUUM t;":                                     true,
		"SET LOCAL statement_timeout = 0;":              true,
		"SELECT set_config('lock_timeout', '0', true);": true,
		"CREATE FUNCTION f() RETURNS trigger LANGUAGE plpgsql AS $$\nBEGIN\n    RETURN NULL;\nEND;\n$$;":         false,
		"CREATE FUNCTION f() RETURNS void LANGUAGE plpgsql AS $body$ BEGIN COMMIT; END; $body$;":                 false,
		"-- COMMIT, then VACUUM\nSELECT 'commit; vacuum', \"end; x\" FROM t;":                                    false,
		"UPDATE t SET a = CASE WHEN b THEN 1 ELSE 2\nEND;":                                                       false,
		"REFRESH MATERIALIZED VIEW CONCURRENTLY v;":                                                              false,
		"/* COMMIT; */ SELECT 1; /* nested /* VACUUM; */ still a comment; */":                                    false,
		"SELECT E'it\\'s; COMMIT';":                                                                              false,
		"CREATE FUNCTION f(a int) RETURNS int LANGUAGE sql BEGIN ATOMIC SELECT CASE WHEN a > 0 THEN 1 END; END;": false,
		"CREATE FUNCTION f() RETURNS int LANGUAGE sql BEGIN ATOMIC SELECT 1; END;\nCOMMIT;":                      true,
		"REINDEX TABLE CONCURRENTLY t;":                                                                          true,
		"ALTER TABLE t DETACH PARTITION p CONCURRENTLY;":                                                         true,
		"START TRANSACTION;":            true,
		"SELECT '$$statement_timeout';": true,
		"SELECT 'unterminated;":         true,
		"SELECT 1 /* unterminated":      true,
	} {
		if problem := migrationTransactionProblem(sql); (problem != "") != refused {
			t.Errorf("migrationTransactionProblem(%q) = %q, want refused %t", sql, problem, refused)
		}
	}
}

// migrationTimeoutName matches a session setting that would replace the
// migration budget.
var migrationTimeoutName = regexp.MustCompile(`(?i)\b(?:statement|lock|idle_in_transaction_session|idle_session|transaction)_timeout\b`)

// migrationTransactionProblem describes what in sql would end the migrator's
// transaction, fail inside it or change its budget, or returns "". It reads
// the top-level statements as psql splits them (explaintest.SplitStatements),
// so comments, string constants and routine bodies hide nothing and end no
// statement; text it cannot split is a problem, never a pass. A timeout
// setting is found in any word or value, routine bodies included.
func migrationTransactionProblem(sql string) string {
	statements, err := explaintest.SplitStatements(sql)
	if err != nil {
		return "cannot be read: " + err.Error()
	}
	for _, statement := range statements {
		for _, text := range slices.Concat(statement.Words, statement.Values) {
			if name := migrationTimeoutName.FindString(text); name != "" {
				return "names the timeout setting " + name
			}
		}
		words := statement.Words
		if len(words) == 0 {
			continue
		}
		switch words[0] {
		case "begin", "commit", "end", "rollback", "abort", "savepoint", "release", "vacuum":
			return "has a statement that cannot run inside the migration transaction: " + words[0]
		case "start", "prepare":
			if len(words) > 1 && words[1] == "transaction" {
				return "has a statement that cannot run inside the migration transaction: " + words[0] + " transaction"
			}
		}
		concurrently := slices.Contains(words, "concurrently")
		for i := range words {
			switch {
			case concurrently && words[i] == "reindex",
				i+1 < len(words) && words[i] == "index" && words[i+1] == "concurrently",
				concurrently && i+1 < len(words) && words[i] == "detach" && words[i+1] == "partition":
				return "has a statement that cannot run inside the migration transaction: " + strings.Join(words, " ")
			}
		}
	}
	return ""
}

func TestCompareMigrationLedgerClassifiesPendingAndDrift(t *testing.T) {
	first := migration{version: 1, name: "000001_first.sql", checksum: sha256.Sum256([]byte("first"))}
	second := migration{version: 2, name: "000002_second.sql", checksum: sha256.Sum256([]byte("second"))}
	available := []migration{first, second}
	validFirst := appliedMigration{name: first.name, checksum: first.checksum}
	validSecond := appliedMigration{name: second.name, checksum: second.checksum}
	for _, test := range []struct {
		name         string
		applied      map[int64]appliedMigration
		current      int64
		pending      int
		migrationErr error
	}{
		{name: "empty", applied: map[int64]appliedMigration{}, pending: 2},
		{name: "pending", applied: map[int64]appliedMigration{1: validFirst}, current: 1, pending: 1},
		{name: "exact", applied: map[int64]appliedMigration{1: validFirst, 2: validSecond}, current: 2},
		{name: "unknown", applied: map[int64]appliedMigration{1: validFirst, 3: {name: "000003_future.sql"}}, migrationErr: ErrMigrationDrift},
		{name: "renamed", applied: map[int64]appliedMigration{1: {name: "renamed.sql", checksum: first.checksum}}, migrationErr: ErrMigrationDrift},
		{name: "checksum drift", applied: map[int64]appliedMigration{1: {name: first.name, checksum: second.checksum}}, migrationErr: ErrMigrationDrift},
	} {
		t.Run(test.name, func(t *testing.T) {
			state, err := compareMigrationLedger(available, test.applied)
			if !errors.Is(err, test.migrationErr) || (test.migrationErr == nil &&
				(state.currentVersion != test.current || len(state.pending) != test.pending)) {
				t.Fatalf("state=%+v error=%v, want current=%d pending=%d error=%v",
					state, err, test.current, test.pending, test.migrationErr)
			}
		})
	}
}

func TestInvalidDatabaseConfigurationIsRedacted(t *testing.T) {
	t.Parallel()

	const secret = "not-a-real-secret"
	pool, err := OpenPool(context.Background(), "://"+secret, PoolOptions{})
	if pool != nil {
		pool.Close()
		t.Fatal("OpenPool returned a pool for an invalid URL")
	}
	if !errors.Is(err, ErrInvalidDatabaseConfiguration) {
		t.Fatalf("OpenPool error = %v", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("OpenPool error exposed database configuration: %v", err)
	}
}
