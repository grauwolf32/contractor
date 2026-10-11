package auditstore

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

var (
	auditEventInsert = regexp.MustCompile(`(?i)\binsert\s+into\s+audit_events\b`)
	// migrationFunction matches one dollar-quoted function definition.
	migrationFunction = regexp.MustCompile(
		`(?is)\bcreate\s+(?:or\s+replace\s+)?function\s+([a-z_][a-z0-9_]*)\s*\(.*?\$\$(.*?)\$\$`,
	)
	eventReservation = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\.(?:advance|created)\(\)`)
	eventAppend      = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\.appended\(\)`)
	eventCounterSet  = regexp.MustCompile(`\bnext_event_sequence\s*=`)
	// An unnamed batch could reserve without appending unnoticed.
	unnamedEventBatch = regexp.MustCompile(`\)\s*\.(?:advance|created|appended)\(\)`)
)

// TestAuditEventsHaveOneWriter scans the Go sources and the effective
// migration schema: INSERT INTO audit_events appears exactly once, in the
// allocator function, and Go code reserves and appends events only through
// auditEventBatch.
func TestAuditEventsHaveOneWriter(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var writers []string
	for _, dir := range []string{"cmd", "internal"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			source, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			relative, _ := filepath.Rel(root, path)
			for range auditEventInsert.FindAllIndex(source, -1) {
				writers = append(writers, relative)
			}
			if strings.Contains(string(source), "contractor_append_audit_events") &&
				relative != filepath.Join("internal", "auditstore", "event_allocation.go") {
				t.Errorf("%s calls the Audit event allocator outside auditEventBatch", relative)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	writers = append(writers, effectiveMigrationEventWriters(t, filepath.Join(root, "internal", "persistence", "migrations"))...)
	if len(writers) != 1 || writers[0] != "function contractor_append_audit_events" {
		t.Fatalf("INSERT INTO audit_events appears in %q, want only function contractor_append_audit_events", writers)
	}
}

// effectiveMigrationEventWriters returns the migration statements that insert
// Audit events once every migration applied: the last definition of each
// function, plus statements outside function bodies.
func effectiveMigrationEventWriters(t *testing.T, dir string) []string {
	t.Helper()
	names, err := filepath.Glob(filepath.Join(dir, "*.sql"))
	if err != nil || len(names) == 0 {
		t.Fatalf("migrations = (%q, %v)", names, err)
	}
	sort.Strings(names)
	functions := map[string]string{}
	var writers []string
	for _, name := range names {
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, match := range migrationFunction.FindAllSubmatch(source, -1) {
			functions[strings.ToLower(string(match[1]))] = string(match[2])
		}
		outside := migrationFunction.ReplaceAll(source, nil)
		for range auditEventInsert.FindAllIndex(outside, -1) {
			writers = append(writers, filepath.Base(name))
		}
	}
	for name, body := range functions {
		for range auditEventInsert.FindAllStringIndex(body, -1) {
			writers = append(writers, "function "+name)
		}
	}
	sort.Strings(writers)
	return writers
}

// TestAuditEventReservationsAreAppended pins the static half of the density
// rule: in this package only auditEventBatch writes next_event_sequence, and
// every batch whose advance() or created() reserves events in a file also
// appends them there.
func TestAuditEventReservationsAreAppended(t *testing.T) {
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	reservations := 0
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if name != "event_allocation.go" && eventCounterSet.Match(source) {
			t.Errorf("%s writes next_event_sequence outside auditEventBatch", name)
		}
		if unnamedEventBatch.Match(source) {
			t.Errorf("%s uses an unnamed Audit event batch; name it so its reservation can be checked", name)
		}
		reserved := map[string]int{}
		for _, match := range eventReservation.FindAllSubmatch(source, -1) {
			reserved[string(match[1])]++
			reservations++
		}
		for _, match := range eventAppend.FindAllSubmatch(source, -1) {
			reserved[string(match[1])]--
		}
		for batch, unmatched := range reserved {
			if unmatched != 0 {
				t.Errorf("%s: %s.advance() and %s.appended() differ by %d", name, batch, batch, unmatched)
			}
		}
	}
	if reservations == 0 {
		t.Fatal("no Audit event reservation found; the scan matched nothing")
	}
}
