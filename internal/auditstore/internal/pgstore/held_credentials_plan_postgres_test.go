package pgstore

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// The held-Audit fixture is an Audit history of heldAudits Audits over six
// owners' Projects. One Audit in heldEvery is running and holds its baseline
// credentials, one in 97 is a draft without a baseline, and the rest have
// completed and released their hold. A baseline names LLM credentials
// llm-<i mod 30> and llm-<(i+7) mod 30> and Runtime credential
// runtime-<i mod 30>, and carries about 8 KB of other content, so it is
// stored out of line as a real baseline is.
const (
	heldAudits      = 3000
	heldEvery       = 50
	heldPadding     = 250
	heldCredentials = 30
)

// heldAuditsProjectsSQL creates the Projects the Audits belong to.
const heldAuditsProjectsSQL = `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
SELECT 'project-' || owner || '-' || project, 'owner-' || owner, 'project', 'Held fixture',
       'create-' || project, 'sha256:' || repeat('a', 64)
FROM generate_series(0, 5) AS owner, generate_series(0, 4) AS project`

// heldAuditsSQL adds Audits $1 to $2 as the fixture describes, with $3
// padding digests in each baseline and one Audit in $4 held.
const heldAuditsSQL = `
INSERT INTO audits (
    audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, baseline_snapshot, state, dispatch_state, hold_state,
    max_rounds, batch_size, max_items_per_round, max_items_total, max_submitted_runs,
    max_item_run_attempts, max_evidence_bytes, created_at, updated_at, started_at, finished_at, phase
)
SELECT 'audit-' || lpad(i::text, 5, '0'), 'owner-' || (i % 6), 'project-' || (i % 6) || '-' || ((i / 6) % 5),
       'profile', '1', 'sha256:' || repeat('b', 64), '{}'::jsonb, '{}'::jsonb,
       CASE WHEN audit.hold <> 'pending' THEN jsonb_build_object(
           'llmCredentialIds', jsonb_build_array('llm-' || (i % 30), 'llm-' || ((i + 7) % 30)),
           'runtimeCredentialIds', jsonb_build_array('runtime-' || (i % 30)),
           'skills', (SELECT jsonb_agg(md5(i::text || '-' || j)) FROM generate_series(1, $3::integer) AS j)) END,
       audit.state, CASE WHEN audit.hold = 'released' THEN 'closed' ELSE 'open' END, audit.hold,
       1, 1, 1, 1, 1, 1, 1024, audit.created_at, audit.created_at,
       CASE WHEN audit.state <> 'draft' THEN audit.created_at END,
       CASE WHEN audit.state = 'completed' THEN audit.created_at + interval '1 hour' END,
       CASE WHEN audit.state = 'draft' THEN 'not-started' ELSE 'rounds' END
FROM generate_series($1::integer, $2::integer) AS i
CROSS JOIN LATERAL (
    SELECT CASE WHEN i % $4::integer = 0 THEN 'held' WHEN i % 97 = 1 THEN 'pending' ELSE 'released' END AS hold,
           timestamptz '2026-09-01 00:00:00+00' + i * interval '1 minute' AS created_at
) AS placed
CROSS JOIN LATERAL (
    SELECT placed.hold, placed.created_at,
           CASE placed.hold WHEN 'held' THEN 'active' WHEN 'pending' THEN 'draft' ELSE 'completed' END AS state
) AS audit`

// heldAuditIDs returns, oldest first, the Audits from first to last that the
// fixture holds with credential under member when one in every Audits is held.
func heldAuditIDs(member, credential string, first, last, every int) []string {
	var held []string
	for i := first; i <= last; i++ {
		if i%every != 0 {
			continue
		}
		names := []string{fmt.Sprintf("runtime-%d", i%heldCredentials)}
		if member == runtimeconfig.LLMCredentialIDsMember {
			names = []string{fmt.Sprintf("llm-%d", i%heldCredentials), fmt.Sprintf("llm-%d", (i+7)%heldCredentials)}
		}
		if slices.Contains(names, credential) {
			held = append(held, fmt.Sprintf("audit-%05d", i))
		}
	}
	return held
}

// TestPostgresHeldAuditCredentialChecksReadOnlyMatchingHeldAudits checks the
// held-Audit credential lookup of managed credential deletion (AUD-R16) on a
// realistic Audit history. It lists exactly the held Audits naming the
// credential under its own kind, oldest first and up to the limit, follows an
// Audit whose hold is released or taken, and reads through the partial index
// of its kind only the held Audits that name the credential, in its custom
// and generic plans: of audits it reads only the entries the index returns for
// the matching held Audits (explaintest.Reads). The plan without the indexes
// is recorded for comparison.
func TestPostgresHeldAuditCredentialChecksReadOnlyMatchingHeldAudits(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	if _, err := pool.Exec(ctx, heldAuditsProjectsSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, heldAuditsSQL, 1, heldAudits, heldPadding, heldEvery); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE audits`); err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	llm, runtime := runtimeconfig.LLMCredentialIDsMember, runtimeconfig.RuntimeCredentialIDsMember
	lists := map[string]func(context.Context, string, int) ([]string, error){
		llm: store.ListHeldAuditIDsByLLMCredential, runtime: store.ListHeldAuditIDsByRuntimeCredential,
	}
	expected := func(member, credential string, limit int) []string {
		held := heldAuditIDs(member, credential, 1, heldAudits, heldEvery)
		return held[:min(limit, len(held))]
	}
	if held := expected(llm, "llm-10", 128); len(held) < 10 {
		t.Fatalf("the fixture holds %d Audits naming llm-10, want enough to page", len(held))
	}
	for _, check := range []struct {
		member, credential string
		limit              int
	}{
		{llm, "llm-10", 128}, {llm, "llm-10", 5}, {llm, "llm-17", 128}, {llm, "runtime-10", 128}, {llm, "llm-missing", 128},
		{runtime, "runtime-10", 128}, {runtime, "runtime-10", 1}, {runtime, "llm-10", 128},
	} {
		held, err := lists[check.member](ctx, check.credential, check.limit)
		if want := expected(check.member, check.credential, check.limit); err != nil || !slices.Equal(held, want) {
			t.Errorf("%s holds of %s up to %d = (%v, %v), want %v", check.member, check.credential, check.limit, held, err, want)
		}
	}

	indexes := map[string]string{llm: "audits_held_llm_credentials_idx", runtime: "audits_held_runtime_credentials_idx"}
	for _, check := range []struct{ member, credential string }{
		{llm, "llm-10"}, {llm, "llm-missing"}, {runtime, "runtime-10"},
	} {
		sql := heldAuditIDsByCredentialSQL(check.member)
		analysis, err := explaintest.Explain(ctx, pool, sql, check.credential, 128)
		if err != nil {
			t.Fatal(err)
		}
		plan, matching := analysis.Plan, int64(len(expected(check.member, check.credential, 128)))
		if !plan.Uses(indexes[check.member]) {
			t.Errorf("%s holds of %s do not find the held Audits through %s:\n%s",
				check.member, check.credential, indexes[check.member], plan)
		}
		// The partial index of the member returns one entry for each held
		// Audit that names the credential, and nothing else reads an Audit.
		// The fixture has only inserted rows so far, so no index holds an
		// entry of a dead row, and no statement compares an indexed column
		// with a value the planner probes an index for.
		if read := analysis.Reads.Of("audits"); read > matching {
			t.Errorf("%s holds of %s read %d Audits for %d matching held Audits:\n%s",
				check.member, check.credential, read, matching, plan)
		}
		original := explainWithoutHeldCredentialIndexes(t, ctx, pool, sql, check.credential)
		t.Logf("%s holds of %s, %d matching: reads %d, %d blocks:\n%s\nwithout the held credential indexes, %d blocks:\n%s",
			check.member, check.credential, matching, analysis.Reads.Of("audits"), plan.Blocks(), plan, original.Blocks(), original)
		if plan.Blocks() >= original.Blocks() {
			t.Errorf("%s holds of %s read %d blocks, no fewer than the %d they read without the held credential indexes",
				check.member, check.credential, plan.Blocks(), original.Blocks())
		}
	}
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	for member, index := range indexes {
		generic, err := explaintest.ExplainGeneric(ctx, conn.Conn().PgConn(), heldAuditIDsByCredentialSQL(member))
		if err != nil {
			t.Fatal(err)
		}
		if !generic.Uses(index) || slices.Contains(generic.SeqScans(), "audits") {
			t.Errorf("the generic plan of %s holds does not use %s:\n%s", member, index, generic)
		}
	}

	// A released hold leaves the index and a new hold enters it. The plans
	// are measured first: the rolled-back hold leaves an entry in each held
	// credential index until a VACUUM removes it.
	all := expected(llm, "llm-10", 128)
	err = pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `UPDATE audits SET hold_state = 'released', dispatch_state = 'closed' WHERE audit_id = $1`,
			all[0]); err != nil {
			return err
		}
		const taken = 3010 // names llm-10
		if _, err := tx.Exec(ctx, heldAuditsSQL, taken, taken, heldPadding, 1); err != nil {
			return err
		}
		want := append(slices.Clone(all[1:]), heldAuditIDs(llm, "llm-10", taken, taken, 1)...)
		if held, err := newTestStore(tx).ListHeldAuditIDsByLLMCredential(ctx, "llm-10", 128); err != nil || !slices.Equal(held, want) {
			t.Errorf("holds after a release and a new hold = (%v, %v), want %v", held, err, want)
		}
		return errHeldRollback
	})
	if !errors.Is(err, errHeldRollback) {
		t.Fatal(err)
	}
}

var errHeldRollback = errors.New("rolled back")

// explainWithoutHeldCredentialIndexes explains sql as the schema before
// V407-025 ran it, in a transaction that drops the held credential indexes,
// if they exist, and is rolled back. Without the indexes the plan checks
// above fail; this comparison then only records the same plan twice.
func explainWithoutHeldCredentialIndexes(
	t *testing.T, ctx context.Context, pool interface {
		Begin(context.Context) (pgx.Tx, error)
	}, sql, credential string,
) explaintest.Node {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `DROP INDEX IF EXISTS audits_held_llm_credentials_idx, audits_held_runtime_credentials_idx`); err != nil {
		t.Fatal(err)
	}
	analysis, err := explaintest.Explain(ctx, tx, sql, credential, 128)
	if err != nil {
		t.Fatal(err)
	}
	return analysis.Plan
}
