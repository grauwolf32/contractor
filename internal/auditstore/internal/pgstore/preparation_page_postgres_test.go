package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// pageStatementTx is a caller's transaction that counts the statements a
// store sends through it and can fail the statement that contains failOn
// with failure.
type pageStatementTx struct {
	pgx.Tx
	statements int
	failOn     string
	failure    error
}

func (tx *pageStatementTx) Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error) {
	tx.statements++
	return tx.Tx.Exec(ctx, sql, arguments...)
}

func (tx *pageStatementTx) Query(ctx context.Context, sql string, arguments ...any) (pgx.Rows, error) {
	tx.statements++
	if tx.failure != nil && strings.Contains(sql, tx.failOn) {
		return nil, tx.failure
	}
	return tx.Tx.Query(ctx, sql, arguments...)
}

func (tx *pageStatementTx) QueryRow(ctx context.Context, sql string, arguments ...any) pgx.Row {
	tx.statements++
	return tx.Tx.QueryRow(ctx, sql, arguments...)
}

func TestListPreparedPageRejectsUnboundedSelections(t *testing.T) {
	ids := func(count int) []string {
		result := make([]string, count)
		for index := range result {
			result[index] = fmt.Sprintf("audit-%03d", index)
		}
		return result
	}
	for name, test := range map[string]struct {
		owner  string
		audits []string
	}{
		"empty page":       {"owner", nil},
		"oversized page":   {"owner", ids(pagelimit.MaxFetchRows + 1)},
		"duplicate Audit":  {"owner", append(ids(2), "audit-000")},
		"invalid Audit ID": {"owner", []string{"../audit"}},
		"missing owner":    {" ", ids(1)},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := listPreparationPage(t.Context(), nil, test.owner, test.audits); !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("selection error = %v", err)
			}
		})
	}
	// Invalid requests fail before the store begins a transaction on its
	// (here absent) database.
	for name, params := range map[string]auditstore.ListParams{
		"list without owner": {OwnerID: " ", Limit: 1},
		"oversized fetch":    {OwnerID: "owner", Limit: pagelimit.MaxFetchRows + 1},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := NewQueryStore(nil).ListPreparedPage(t.Context(), params); !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("page error = %v", err)
			}
		})
	}
}

// TestSharedPostgresListPreparedPageReadsAFullFetchInTwoStatements checks the
// list page read on a full fetch of preparing Audits against the detail read
// of each Audit: the list and one preparation statement, the same roles and
// latest attempts, no preparation statement for a page without preparing
// Audits, and storage failures keeping their class. It also counts the
// statements of the detail read (ASC-R149).
func TestSharedPostgresListPreparedPageReadsAFullFetchInTwoStatements(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	const owner, projectID = "owner-preparation-page", "project-preparation-page"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: "Preparation page",
		IdempotencyKey: "project", RequestDigest: testDigest("1"),
	}); err != nil {
		t.Fatal(err)
	}
	profile, snapshot := preparationPageProfile(t)
	store := newTestStore(pool)
	inputs := map[string]auditstore.ExactArtifact{"source": testExact("inputs", "source", "source-r1"),
		"settings": testExact("inputs", "settings", "settings-r1")}
	settings := inputs["settings"]
	settings.MediaType = "application/json"
	inputs["settings"] = settings
	baseline, err := json.Marshal(map[string]any{"schema": "contractor.audit.baseline.v1", "inputs": inputs})
	if err != nil {
		t.Fatal(err)
	}
	create := func(auditID string) auditstore.Audit {
		t.Helper()
		draft, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: owner, ProjectID: projectID,
			Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
			ProfileSnapshot: snapshot, InputSelection: json.RawMessage(`{}`),
			Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 16, MaxItemsTotal: 16,
				MaxSubmittedRuns: 50, MaxItemRunAttempts: 3, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey: "create-" + auditID, RequestDigest: testDigest("2"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return draft
	}
	start := func(auditID string) auditstore.Audit {
		t.Helper()
		draft := create(auditID)
		started, _, err := store.StartPreparation(ctx, auditstore.StartPreparationParams{
			OwnerID: owner, AuditID: auditID, ExpectedRevision: draft.Revision, BaselineSnapshot: baseline,
			DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start-" + auditID, RequestDigest: testDigest("3"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return started
	}
	// The oldest Audit is a draft: it pins the same prepare roles but is not
	// preparing, and the full fetch below leaves it on the next page.
	create("audit-draft")
	// Three Audits leave the generate-api role running, retrying after a
	// failed attempt and failed after exhausting its two attempts.
	attempted := []auditstore.Audit{start("audit-running"), start("audit-retrying"), start("audit-exhausted")}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "page-controller", Lease: 5 * time.Minute, Limit: len(attempted)})
	if err != nil || len(claims) != len(attempted) {
		t.Fatalf("claim attempted Audits = (%+v, %v)", claims, err)
	}
	claimOf := make(map[string]auditstore.ControllerClaim, len(claims))
	for _, claim := range claims {
		claimOf[claim.AuditID] = claim
	}
	attempt := func(auditID string, number int, collect bool) {
		t.Helper()
		claim := claimOf[auditID]
		executionID := auditdomain.DeterministicID("prepare", auditID, "generate-api", fmt.Sprint(number))
		execution, _, err := store.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
			Claim: claim, ExecutionID: executionID, Role: auditstore.ExecutionPrepare, WorkflowRole: "generate-api",
			LaneClass: resolved.LaneClassModel, RoleAttempt: &number, Manifest: testExact(auditdomain.ArtifactNamespace(auditID), "manifest", "manifest-r1"),
			SubmissionKey: executionID, RequestDigest: testDigest("4"),
			Preparation: &auditstore.PreparationSnapshot{Inputs: map[string]auditstore.ExactArtifact{"source": inputs["source"]}, Parameters: map[string]string{}},
		})
		if err != nil || !collect {
			if err != nil {
				t.Fatal(err)
			}
			return
		}
		runID := "run-" + executionID
		if _, err := insertAndBindTestRun(t, ctx, pool, claim, execution, runID, owner, projectID); err != nil {
			t.Fatal(err)
		}
		generation, sequence := terminateTestRun(t, ctx, pool, runID, "failed")
		if _, err := store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{
			Claim: claim, ExecutionID: executionID, RunID: runID, Generation: generation, Sequence: sequence,
		}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := store.Collect(ctx, auditstore.CollectParams{
			Claim: claim, ExecutionID: executionID, ReceiptID: "receipt-" + executionID,
			RequestDigest: testDigest("5"), Disposition: auditstore.CollectionExecutionFailed,
		}); err != nil {
			t.Fatal(err)
		}
	}
	attempt("audit-running", 1, false)
	attempt("audit-retrying", 1, true)
	attempt("audit-exhausted", 1, true)
	attempt("audit-exhausted", 2, true)

	for index := len(attempted); index < pagelimit.MaxFetchRows; index++ {
		start(fmt.Sprintf("audit-pending-%03d", index))
	}

	params := auditstore.ListParams{OwnerID: owner, ProjectID: new(projectID), Limit: pagelimit.MaxFetchRows}
	// read lists a page in a caller's snapshot transaction that counts the
	// store's statements and fails the one containing failOn with failure.
	read := func(t *testing.T, params auditstore.ListParams, failOn string, failure error) (
		[]auditstore.Audit, map[string][]auditstore.ListedPreparationRole, int, error,
	) {
		t.Helper()
		tx, err := pool.BeginTx(ctx, snapshotTransaction)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
		counted := &pageStatementTx{Tx: tx, failOn: failOn, failure: failure}
		audits, roles, err := NewQueryStore(counted).ListPreparedPage(ctx, params)
		return audits, roles, counted.statements, err
	}
	audits, roles, statements, err := read(t, params, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if statements != 2 || len(audits) != pagelimit.MaxFetchRows || len(roles) != pagelimit.MaxFetchRows {
		t.Fatalf("full fetch took %d statements for %d Audits and %d preparations", statements, len(audits), len(roles))
	}
	statuses := map[string]auditdomain.PreparationStatus{}
	for _, audit := range audits {
		// Every listed Audit has the roles its detail reads (ASC-R150).
		detail, err := NewQueryStore(pool).GetPreparedDetail(ctx, owner, audit.AuditID, false)
		if err != nil || detail.Audit.Revision != audit.Revision {
			t.Fatalf("%s detail = revision %d, %v", audit.AuditID, detail.Audit.Revision, err)
		}
		want := detail.Roles
		executions, err := store.ListPreparationExecutions(ctx, audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		got := roles[audit.AuditID]
		if len(got) != len(want) || len(got) != 2 {
			t.Fatalf("%s roles = %+v, want %+v", audit.AuditID, got, want)
		}
		for index, role := range got {
			if !reflect.DeepEqual(role, want[index]) {
				t.Fatalf("%s role %d = %+v, want %+v", audit.AuditID, index, role, want[index])
			}
			if (role.Latest == nil) != (role.ExecutionID == nil) {
				t.Fatalf("%s role %s latest attempt = %+v", audit.AuditID, role.WorkflowRole, role.Latest)
			}
			for _, execution := range executions {
				if role.Latest != nil && execution.ExecutionID == role.Latest.ExecutionID &&
					(*execution.RoleAttempt != role.Latest.RoleAttempt || !reflect.DeepEqual(execution.RunID, role.Latest.RunID) ||
						execution.State != role.Latest.State || len(role.Latest.Outputs) != 0) {
					t.Fatalf("%s latest attempt = %+v, want %+v", audit.AuditID, role.Latest, execution)
				}
			}
		}
		statuses[audit.AuditID] = got[0].Status
	}
	if statuses["audit-running"] != auditdomain.PreparationRunning || statuses["audit-retrying"] != auditdomain.PreparationRunning ||
		statuses["audit-exhausted"] != auditdomain.PreparationFailed || statuses["audit-pending-199"] != auditdomain.PreparationPending ||
		roles["audit-exhausted"][0].Attempts != 2 || roles["audit-exhausted"][0].Latest.RoleAttempt != 2 {
		t.Fatalf("page statuses = %+v", statuses)
	}

	// A pool-backed store reads the same page in its own snapshot.
	pooled, pooledRoles, err := NewQueryStore(pool).ListPreparedPage(ctx, params)
	if err != nil || !reflect.DeepEqual(pooled, audits) || !reflect.DeepEqual(pooledRoles, roles) {
		t.Fatalf("pool-backed page differs: %v", err)
	}
	// A page without a preparing or inventory Audit issues only the list
	// statement.
	drafts := params
	drafts.States = []auditstore.AuditState{auditstore.AuditDraft}
	listed, none, statements, err := read(t, drafts, "", nil)
	if err != nil || statements != 1 || len(listed) != 1 || listed[0].AuditID != "audit-draft" || len(none) != 0 {
		t.Fatalf("draft page = %d Audits, %d preparations in %d statements: %v", len(listed), len(none), statements, err)
	}

	// Storage failures keep their class.
	_, _, _, err = read(t, params, "WITH ORDINALITY AS listed",
		&pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure, Message: "could not serialize access"})
	if !persistencepostgres.IsTransactionConflict(err) || errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("transient page failure = %v", err)
	}
	unavailable := errors.New("storage unavailable")
	if _, _, _, err := read(t, params, "WITH ORDINALITY AS listed", unavailable); !errors.Is(err, unavailable) {
		t.Fatalf("unavailable page = %v", err)
	}

	// detail reads one Audit's detail, with its current Round, in a caller's
	// snapshot transaction, as read does a page.
	detail := func(t *testing.T, ownerID, auditID, failOn string, failure error) (auditstore.PreparedDetail, int, error) {
		t.Helper()
		tx, err := pool.BeginTx(ctx, snapshotTransaction)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
		counted := &pageStatementTx{Tx: tx, failOn: failOn, failure: failure}
		read, err := NewQueryStore(counted).GetPreparedDetail(ctx, ownerID, auditID, true)
		return read, counted.statements, err
	}
	// A preparing Audit takes the Audit and the list's preparation statement
	// and has neither inventory nor a Round; a draft takes the Audit
	// statement alone.
	exhausted, statements, err := detail(t, owner, "audit-exhausted", "", nil)
	if err != nil || statements != 2 || !reflect.DeepEqual(exhausted.Roles, roles["audit-exhausted"]) ||
		exhausted.InventoryLink != nil || exhausted.FirstRound != nil || exhausted.CurrentRound != nil || exhausted.CurrentItems != nil {
		t.Fatalf("preparing detail = %+v in %d statements: %v", exhausted, statements, err)
	}
	draft, statements, err := detail(t, owner, "audit-draft", "", nil)
	if err != nil || statements != 1 || draft.Audit.AuditID != "audit-draft" || draft.Roles != nil {
		t.Fatalf("draft detail = %+v in %d statements: %v", draft, statements, err)
	}
	for _, missing := range [][2]string{{"another-owner", "audit-exhausted"}, {owner, "audit-missing"}} {
		if _, _, err := detail(t, missing[0], missing[1], "", nil); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("detail of %s's %s = %v", missing[0], missing[1], err)
		}
	}
	_, _, err = detail(t, owner, "audit-exhausted", "WITH ORDINALITY AS listed",
		&pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure, Message: "could not serialize access"})
	if !persistencepostgres.IsTransactionConflict(err) || errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("transient detail failure = %v", err)
	}
	if _, _, err := detail(t, owner, "audit-exhausted", "WITH ORDINALITY AS listed", unavailable); !errors.Is(err, unavailable) {
		t.Fatalf("unavailable detail = %v", err)
	}
}

// preparationPageProfile resolves the shared prepared OpenAPI scan profile
// with a second prepare role that depends on the first one's output.
func preparationPageProfile(t *testing.T) (auditprofile.ResolvedAuditProfile, json.RawMessage) {
	t.Helper()
	root := configtest.CopyWithPolicies(t, "../../../auditservice/testdata/catalog")
	data := contractstest.ReadFile(t, "api", "testdata", "audit-composition", "prepared-openapi-scan.yaml")
	data = []byte(strings.Replace(string(data), "    scan:\n", `    refine-api:
      kind: prepare
      ref: openapi-from-workspace@7
      maxRunAttempts: 2
      inputs:
        source: {source: audit-input, name: source}
        existing_openapi: {source: prepare-output, role: generate-api, name: api}
      parameters: {}
      outputs: {api: openapi, validation: openapi_validation_report}
    scan:
`, 1))
	if err := os.WriteFile(filepath.Join(root, "audit-profiles", "prepared.yaml"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	return profile, snapshot
}
