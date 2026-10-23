package persistence

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// advisoryLockFiles may take a PostgreSQL advisory lock or hash its key
// directly, each for the reason given. Every other lock is taken through
// persistencepostgres.NewAdvisoryKey and LockAdvisory (PER-R24).
var advisoryLockFiles = map[string]string{
	"internal/persistence/postgres/advisory.go":                      "the one advisory key builder and lock",
	"internal/persistence/postgres/migrate.go":                       "the migration lock, scoped to the schema it migrates",
	"internal/persistence/postgres/budgets_postgres_test.go":         "holds the migration lock to time out a migrator",
	"internal/app/single_active.go":                                  "the session lease of the active Server, whose fixed key every Server version shares (PER-R2)",
	"internal/app/single_active_database_postgres_test.go":           "holds the session lease from another database",
	"internal/httpapi/public/eval_external_records_postgres_test.go": "finds the requests that wait on a receipt key in pg_locks by the key's hash",
	"tests/integration/lease/standby_process_test.go":                "holds the session lease so a second Server process starts as the standby",
}

// receiptHelperFiles are the only production Go files whose SQL reads or
// writes mutation_receipts: the shared receipt helper. Every keyed mutation
// records and replays its receipt through it (PER-R24).
var receiptHelperFiles = map[string]bool{
	"internal/persistence/postgres/receipts.go": true,
}

var (
	advisoryLockCall = regexp.MustCompile(`pg_(?:try_)?advisory_(?:xact_)?(?:un)?lock|hashtext(?:extended)?\s*\(`)
	receiptTable     = regexp.MustCompile(`\bmutation_receipts\b`)
)

// TestKeyedMutationsUseTheSharedHelpers scans the Go sources: an advisory
// lock outside the listed files bypasses the one key builder, and production
// SQL on mutation_receipts outside the helper bypasses the shared receipt
// flow.
func TestKeyedMutationsUseTheSharedHelpers(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	self := "internal/persistence/keyed_mutation_test.go"
	unused := map[string]bool{}
	for name := range advisoryLockFiles {
		unused[name] = true
	}
	var violations []string
	scanned := 0
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			name := entry.Name()
			if path != root && (strings.HasPrefix(name, ".") || name == "node_modules" || name == "testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if relative == self {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		scanned++
		if advisoryLockCall.Match(content) {
			if _, allowed := advisoryLockFiles[relative]; allowed {
				delete(unused, relative)
			} else {
				violations = append(violations, relative+": takes an advisory lock without persistencepostgres.NewAdvisoryKey")
			}
		}
		if !strings.HasSuffix(relative, "_test.go") && !receiptHelperFiles[relative] && receiptTable.Match(content) {
			violations = append(violations, relative+": reads or writes mutation_receipts outside the shared receipt helper")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if scanned < 100 {
		t.Fatalf("scanned %d Go files, the walk missed the repository", scanned)
	}
	for name := range unused {
		violations = append(violations, name+": listed for advisory locks but takes none")
	}
	if len(violations) != 0 {
		t.Fatalf("keyed mutation helpers bypassed:\n%s", strings.Join(violations, "\n"))
	}
}

// TestKeyedMutationScannerReportsBypasses proves the scanner's patterns match
// what they guard.
func TestKeyedMutationScannerReportsBypasses(t *testing.T) {
	for _, sample := range []string{
		"SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
		"SELECT pg_advisory_xact_lock_shared($1::integer, hashtext($2))",
		"SELECT pg_try_advisory_lock($1)",
		"SELECT pg_advisory_unlock($1)",
	} {
		if !advisoryLockCall.MatchString(sample) {
			t.Fatalf("advisory lock not found in %q", sample)
		}
	}
	if !receiptTable.MatchString("INSERT INTO mutation_receipts (key_scope) VALUES ($1)") ||
		receiptTable.MatchString("SELECT * FROM mutation_receipts_view") {
		t.Fatal("receipt table pattern does not match the table name exactly")
	}
}
