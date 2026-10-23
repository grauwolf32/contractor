package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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

// pageStatementDB counts the statements a store sends and can fail the
// statement that contains failOn with failure.
type pageStatementDB struct {
	persistencepostgres.DBTX
	statements int
	failOn     string
	failure    error
}

func (db *pageStatementDB) Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error) {
	db.statements++
	return db.DBTX.Exec(ctx, sql, arguments...)
}

func (db *pageStatementDB) Query(ctx context.Context, sql string, arguments ...any) (pgx.Rows, error) {
	db.statements++
	if db.failure != nil && strings.Contains(sql, db.failOn) {
		return nil, db.failure
	}
	return db.DBTX.Query(ctx, sql, arguments...)
}

func (db *pageStatementDB) QueryRow(ctx context.Context, sql string, arguments ...any) pgx.Row {
	db.statements++
	return db.DBTX.QueryRow(ctx, sql, arguments...)
}

func TestListPreparationPageRejectsUnboundedSelections(t *testing.T) {
	store := NewQueryStore(nil)
	listed := func(count int) []auditstore.ListedAudit {
		result := make([]auditstore.ListedAudit, count)
		for index := range result {
			result[index] = auditstore.ListedAudit{AuditID: fmt.Sprintf("audit-%03d", index), Revision: 1}
		}
		return result
	}
	for name, test := range map[string]struct {
		owner  string
		audits []auditstore.ListedAudit
	}{
		"empty page":       {"owner", nil},
		"oversized page":   {"owner", listed(pagelimit.MaxRows + 1)},
		"duplicate Audit":  {"owner", append(listed(2), auditstore.ListedAudit{AuditID: "audit-000", Revision: 2})},
		"invalid Audit ID": {"owner", []auditstore.ListedAudit{{AuditID: "../audit", Revision: 1}}},
		"zero revision":    {"owner", []auditstore.ListedAudit{{AuditID: "audit", Revision: 0}}},
		"revision too big": {"owner", []auditstore.ListedAudit{{AuditID: "audit", Revision: math.MaxInt64 + 1}}},
		"missing owner":    {" ", listed(1)},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := store.ListPreparationPage(t.Context(), test.owner, test.audits); !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("selection error = %v", err)
			}
		})
	}
}

// TestSharedPostgresListPreparationPageReadsAFullPageInOneFencedStatement checks the
// list hydration read against the per-Audit preparation reads on a full page
// of preparing Audits: one statement, the same roles and latest attempts, and
// a conflict whenever a listed Audit is not the one the page read.
func TestSharedPostgresListPreparationPageReadsAFullPageInOneFencedStatement(t *testing.T) {
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
	start := func(auditID string) auditstore.Audit {
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
		started, _, err := store.StartPreparation(ctx, auditstore.StartPreparationParams{
			OwnerID: owner, AuditID: auditID, ExpectedRevision: draft.Revision, BaselineSnapshot: baseline,
			DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start-" + auditID, RequestDigest: testDigest("3"),
		})
		if err != nil {
			t.Fatal(err)
		}
		return started
	}
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

	page := make([]auditstore.ListedAudit, 0, pagelimit.MaxRows)
	for _, audit := range attempted {
		current, err := store.Get(ctx, owner, audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		page = append(page, auditstore.ListedAudit{AuditID: current.AuditID, Revision: current.Revision})
	}
	for index := len(page); index < pagelimit.MaxRows; index++ {
		audit := start(fmt.Sprintf("audit-pending-%03d", index))
		page = append(page, auditstore.ListedAudit{AuditID: audit.AuditID, Revision: audit.Revision})
	}

	counted := &pageStatementDB{DBTX: pool}
	roles, err := NewQueryStore(counted).ListPreparationPage(ctx, owner, page)
	if err != nil {
		t.Fatal(err)
	}
	if counted.statements != 1 || len(roles) != pagelimit.MaxRows {
		t.Fatalf("full page took %d statements for %d Audits", counted.statements, len(roles))
	}
	statuses := map[string]auditdomain.PreparationStatus{}
	for _, listed := range page {
		want, err := store.ListPreparationRoles(ctx, listed.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		executions, err := store.ListPreparationExecutions(ctx, listed.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		got := roles[listed.AuditID]
		if len(got) != len(want) || len(got) != 2 {
			t.Fatalf("%s roles = %+v, want %+v", listed.AuditID, got, want)
		}
		for index, role := range got {
			if !reflect.DeepEqual(role.PreparationRole, want[index]) {
				t.Fatalf("%s role %d = %+v, want %+v", listed.AuditID, index, role.PreparationRole, want[index])
			}
			if (role.Latest == nil) != (role.ExecutionID == nil) {
				t.Fatalf("%s role %s latest attempt = %+v", listed.AuditID, role.WorkflowRole, role.Latest)
			}
			for _, execution := range executions {
				if role.Latest != nil && execution.ExecutionID == role.Latest.ExecutionID &&
					(*execution.RoleAttempt != role.Latest.RoleAttempt || !reflect.DeepEqual(execution.RunID, role.Latest.RunID) ||
						execution.State != role.Latest.State || len(role.Latest.Outputs) != 0) {
					t.Fatalf("%s latest attempt = %+v, want %+v", listed.AuditID, role.Latest, execution)
				}
			}
		}
		statuses[listed.AuditID] = got[0].Status
	}
	if statuses["audit-running"] != auditdomain.PreparationRunning || statuses["audit-retrying"] != auditdomain.PreparationRunning ||
		statuses["audit-exhausted"] != auditdomain.PreparationFailed || statuses["audit-pending-199"] != auditdomain.PreparationPending ||
		roles["audit-exhausted"][0].Attempts != 2 || roles["audit-exhausted"][0].Latest.RoleAttempt != 2 {
		t.Fatalf("page statuses = %+v", statuses)
	}

	// Any listed Audit that is not the one the page read fails the whole page.
	for name, change := range map[string]func([]auditstore.ListedAudit) (string, []auditstore.ListedAudit){
		"older revision": func(page []auditstore.ListedAudit) (string, []auditstore.ListedAudit) {
			page[7].Revision--
			return owner, page
		},
		"newer revision": func(page []auditstore.ListedAudit) (string, []auditstore.ListedAudit) {
			page[199].Revision++
			return owner, page
		},
		"absent Audit": func(page []auditstore.ListedAudit) (string, []auditstore.ListedAudit) {
			page[0].AuditID = "audit-absent"
			return owner, page
		},
		"other owner": func(page []auditstore.ListedAudit) (string, []auditstore.ListedAudit) {
			return "another-owner", page
		},
	} {
		t.Run(name, func(t *testing.T) {
			changedOwner, changed := change(append([]auditstore.ListedAudit(nil), page...))
			if _, err := NewQueryStore(pool).ListPreparationPage(ctx, changedOwner, changed); !errors.Is(err, auditstore.ErrConflict) {
				t.Fatalf("changed page = %v, want conflict", err)
			}
		})
	}
	// An owner pause after the page was read advances one listed revision.
	if _, _, err := store.Transition(ctx, auditstore.TransitionParams{
		OwnerID: owner, AuditID: page[42].AuditID, ExpectedRevision: page[42].Revision,
		ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
		IdempotencyKey: "pause-listed", RequestDigest: testDigest("6"),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := NewQueryStore(pool).ListPreparationPage(ctx, owner, page); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("page after a concurrent revision = %v, want conflict", err)
	}

	// Storage failures keep their class rather than becoming a conflict.
	transient := &pageStatementDB{DBTX: pool, failOn: "WITH ORDINALITY AS listed",
		failure: &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure, Message: "could not serialize access"}}
	if _, err := NewQueryStore(transient).ListPreparationPage(ctx, owner, page[:1]); !persistencepostgres.IsTransactionConflict(err) ||
		errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("transient page failure = %v", err)
	}
	unavailable := errors.New("storage unavailable")
	broken := &pageStatementDB{DBTX: pool, failOn: "WITH ORDINALITY AS listed", failure: unavailable}
	if _, err := NewQueryStore(broken).ListPreparationPage(ctx, owner, page[:1]); !errors.Is(err, unavailable) {
		t.Fatalf("unavailable page = %v", err)
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
	catalog, err := configload.Load(root)
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
