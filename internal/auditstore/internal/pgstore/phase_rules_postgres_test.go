package pgstore

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/jackc/pgx/v5/pgconn"
)

// TestPostgresPhaseRulesFollowPrecedesRounds holds the database's copies of
// the phases before Rounds to auditdomain.AuditPhase.PrecedesRounds, for
// every AuditPhase the Go code declares. SQL cannot call the predicate, so
// the audits_phase_shape constraint and the preparation baseline trigger
// list the phases themselves: the constraint must admit every Go phase and
// require a baseline and no current Round exactly for the predicate's
// phases, and the trigger must treat exactly those phases as begun
// preparation and as needing a prepare baseline.
func TestPostgresPhaseRulesFollowPrecedesRounds(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	phases := declaredAuditPhases(t)

	var definition string
	if err := pool.QueryRow(ctx, `
SELECT pg_get_constraintdef(oid) FROM pg_constraint
 WHERE conrelid = 'audits'::regclass AND conname = 'audits_phase_shape'`).Scan(&definition); err != nil {
		t.Fatal(err)
	}
	expression, found := strings.CutPrefix(definition, "CHECK ")
	if !found {
		t.Fatalf("audits_phase_shape is not a CHECK constraint: %s", definition)
	}
	// Every phase the constraint names is a Go phase.
	for _, literal := range regexp.MustCompile(`'([^']*)'::text`).FindAllStringSubmatch(expression, -1) {
		if !slices.Contains(phases, auditdomain.AuditPhase(literal[1])) {
			t.Errorf("audits_phase_shape names %q, which is no Go AuditPhase", literal[1])
		}
	}
	// accepts evaluates the constraint on one row shape; a NULL result
	// passes a CHECK constraint.
	accepts := func(phase auditdomain.AuditPhase, round *string, baseline bool) bool {
		t.Helper()
		var accepted *bool
		if err := pool.QueryRow(ctx, `SELECT `+expression+` FROM (
SELECT $1::text AS phase, $2::text AS current_round_id,
       CASE WHEN $3 THEN '{}'::jsonb END AS baseline_snapshot) AS audits`,
			string(phase), round, baseline).Scan(&accepted); err != nil {
			t.Fatal(err)
		}
		return accepted == nil || *accepted
	}
	round := "round-1"
	for _, phase := range phases {
		before := phase.PrecedesRounds()
		if !accepts(phase, nil, true) {
			t.Errorf("audits_phase_shape rejects the Go phase %q", phase)
		}
		if accepts(phase, &round, true) == before {
			t.Errorf("audits_phase_shape on %q with a current Round: accepted=%t, but PrecedesRounds=%t", phase, !before, before)
		}
		if accepts(phase, nil, false) == before {
			t.Errorf("audits_phase_shape on %q without a baseline: accepted=%t, but PrecedesRounds=%t", phase, !before, before)
		}
	}
	if accepts("planning", nil, true) {
		t.Error("audits_phase_shape admits a value that is no phase")
	}

	// The trigger function runs on a scratch table with the columns it reads,
	// so no other audits trigger or constraint answers first.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, `
CREATE TEMP TABLE audit_phase_probe (
    audit_id text, owner_id text, project_id text, profile_name text, profile_version text,
    profile_digest text, profile_snapshot jsonb, input_selection jsonb, baseline_snapshot jsonb, phase text
) ON COMMIT DROP;
CREATE TRIGGER audit_phase_probe_preparation BEFORE UPDATE ON audit_phase_probe
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_preparation_baseline();`); err != nil {
		t.Fatal(err)
	}
	// update inserts one Audit row in phase from, applies set to it and
	// returns the trigger's rejection message, or "".
	update := func(id string, from auditdomain.AuditPhase, set string, arguments ...any) string {
		t.Helper()
		savepoint, err := tx.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = savepoint.Rollback(ctx) }()
		if _, err := savepoint.Exec(ctx, `
INSERT INTO audit_phase_probe VALUES ($1, 'owner', 'project', 'profile', '1', 'sha256:0',
    '{"workflows":{}}', '{}', '{"schema":"probe"}', $2)`, id, string(from)); err != nil {
			t.Fatal(err)
		}
		_, err = savepoint.Exec(ctx, `UPDATE audit_phase_probe SET `+set+` WHERE audit_id = $1`, append([]any{id}, arguments...)...)
		var pgError *pgconn.PgError
		switch {
		case err == nil:
			return ""
		case errors.As(err, &pgError) && pgError.Code == "23514":
			return pgError.Message
		}
		t.Fatal(err)
		return ""
	}
	for index, phase := range phases {
		before := phase.PrecedesRounds()
		id := "phase-probe-" + strconv.Itoa(index)
		// A changed baseline is rejected once preparation began.
		if message := update(id, phase, `baseline_snapshot = '{"schema":"changed"}'`); (message == "Audit preparation baseline is immutable") != before {
			t.Errorf("changing the baseline in phase %q: rejection %q, but PrecedesRounds=%t", phase, message, before)
		}
		// Entering a phase before Rounds requires a prepare baseline, which
		// the probe row does not have.
		if message := update(id, auditdomain.AuditPhaseNotStarted, `phase = $2`, string(phase)); (message == "Audit preparation baseline or profile is invalid") != before {
			t.Errorf("entering phase %q without a prepare baseline: rejection %q, but PrecedesRounds=%t", phase, message, before)
		}
	}
}

// declaredAuditPhases returns every AuditPhase constant the auditdomain
// package declares, so a phase added only in Go is checked too.
func declaredAuditPhases(t *testing.T) []auditdomain.AuditPhase {
	t.Helper()
	directory := contractstest.Path(t, "internal", "auditdomain")
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	var phases []auditdomain.AuditPhase
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, entry.Name()), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range file.Decls {
			general, ok := declaration.(*ast.GenDecl)
			if !ok || general.Tok != token.CONST {
				continue
			}
			for _, spec := range general.Specs {
				value := spec.(*ast.ValueSpec)
				if kind, ok := value.Type.(*ast.Ident); !ok || kind.Name != "AuditPhase" {
					continue
				}
				for _, literal := range value.Values {
					text, err := strconv.Unquote(literal.(*ast.BasicLit).Value)
					if err != nil {
						t.Fatal(err)
					}
					phases = append(phases, auditdomain.AuditPhase(text))
				}
			}
		}
	}
	if len(phases) < 4 {
		t.Fatalf("auditdomain declares %d AuditPhase constants: %v", len(phases), phases)
	}
	for _, phase := range phases {
		if !phase.Valid() {
			t.Fatalf("declared AuditPhase %q is not Valid", phase)
		}
	}
	return phases
}
