package evalstore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type evalQueryCounter struct {
	pg.DBTX
	queries int
}

func (db *evalQueryCounter) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	db.queries++
	return db.DBTX.Query(ctx, sql, args...)
}
func (db *evalQueryCounter) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	db.queries++
	return db.DBTX.QueryRow(ctx, sql, args...)
}

func TestPostgresEvalTenThousandSelectedMembersBoundedPages(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "evaluation")
	template := createExperiment(t, pool, scope, "template", "trace-1", "external-workflow")
	var trace struct {
		Items []evaldomain.Pair `json:"items"`
	}
	raw, err := os.ReadFile("../../api/testdata/evals/valid/pair-page.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &trace); err != nil {
		t.Fatal(err)
	}
	// Seed the persistence boundary directly: this measures complete-view
	// publication and pages, independently of portable plan authoring size.
	const id = "large-comparison"
	memberRows := make([][]any, 0, evaldomain.MaxMembers)
	projectionRows := make([][]any, 0, evaldomain.MaxMembers)
	for ordinal := 0; ordinal < evaldomain.MaxMembers; ordinal++ {
		pairNumber := ordinal / 2
		pair := trace.Items[pairNumber%len(trace.Items)]
		v := pair.A
		if ordinal%2 == 1 {
			v = pair.B
		}
		// Avoid sharing fixture pointer fields across samples.
		var cloned evaldomain.MemberView
		if err = json.Unmarshal(bytesOf(v), &cloned); err != nil {
			t.Fatal(err)
		}
		v = cloned
		suite := fmt.Sprintf("suite-%02d", pairNumber/100)
		caseID := fmt.Sprintf("case-%04d", pairNumber)
		memberID, err := evaldomain.MemberID(id, suite, caseID, 1, v.Member.VariantID)
		if err != nil {
			t.Fatal(err)
		}
		pairID, err := evaldomain.PairID(id, suite, caseID, 1)
		if err != nil {
			t.Fatal(err)
		}
		v.Member.ID, v.Member.SuiteID, v.Member.CaseID, v.Member.Sample = memberID, suite, caseID, 1
		if v.Execution.Ref != nil {
			v.Execution.Ref.ID = fmt.Sprintf("run-%d", ordinal)
		}
		if v.Usage != nil {
			for _, measure := range []*evaldomain.Measure{&v.Usage.InputTokens, &v.Usage.OutputTokens, &v.Usage.TotalTokens, &v.Usage.CachedInputTokens, &v.Usage.ModelCalls, &v.Usage.ToolCalls, &v.Usage.ToolFailures, &v.Usage.WallMS} {
				measure.Scope.MemberID = memberID
				if v.Execution.Ref != nil {
					measure.Scope.Executions = []evaldomain.ExecutionRef{*v.Execution.Ref}
				}
			}
		}
		complete := evaldomain.QualityComplete(v)
		memberRows = append(memberRows, []any{id, memberID, pairID, ordinal, suite, caseID, 1, v.Member.VariantID, v.Member.CaseSHA256, v.Member.BindingSHA256, v.Member.Eligibility, "run", []byte("{}"), fmt.Sprintf("large-%d", ordinal)})
		projectionRows = append(projectionRows, []any{memberID, bytesOf(v), complete})
	}
	mustTx(t, pool, func(st *Store) error {
		// Bulk fixture setup has a separate bound; measured page queries retain
		// the normal 10-second timeout.
		if _, err := st.db.Exec(ctx, `SET LOCAL statement_timeout='60s'`); err != nil {
			return err
		}
		_, err := st.db.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms) VALUES($1,$2,$3,$1,'external','Large selected fixture','finished',$4,1,1000)`, id, scope.OwnerID, scope.ProjectID, evaldomain.MaxMembers)
		if err != nil {
			return err
		}
		_, err = st.db.Exec(ctx, `INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,plan_sha256,setup) SELECT $1,document_kind,document,plan_sha256,setup FROM eval_frozen_plans WHERE experiment_id=$2`, id, template.ID)
		if err != nil {
			return err
		}
		if _, err = st.tx.CopyFrom(ctx, pgx.Identifier{"eval_members"}, []string{"experiment_id", "member_id", "pair_id", "ordinal", "suite_id", "case_id", "sample", "variant_id", "case_sha256", "binding_sha256", "eligibility", "execution_kind", "recipe", "submission_key"}, pgx.CopyFromRows(memberRows)); err != nil {
			return err
		}
		_, err = st.db.Exec(ctx, `CREATE TEMP TABLE selected_fixture(member_id text,document bytea,complete boolean) ON COMMIT DROP`)
		if err != nil {
			return err
		}
		if _, err = st.tx.CopyFrom(ctx, pgx.Identifier{"selected_fixture"}, []string{"member_id", "document", "complete"}, pgx.CopyFromRows(projectionRows)); err != nil {
			return err
		}
		_, err = st.db.Exec(ctx, `UPDATE eval_member_projections p SET projected_revision=revision,document=f.document,collection_complete=f.complete FROM selected_fixture f WHERE p.experiment_id=$1 AND p.member_id=f.member_id`, id)
		if err != nil {
			return err
		}
		_, err = st.db.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES($1)`, id)
		return err
	})
	claims, err := NewPostgresStore(pool).Claim(ctx, "large-reader", 5*time.Minute, 10)
	if err != nil {
		t.Fatal(err)
	}
	var claim Claim
	for _, candidate := range claims {
		if candidate.ExperimentID == id {
			claim = candidate
		}
	}
	if claim.ExperimentID == "" {
		t.Fatal("large fixture was not claimable")
	}
	var view *View
	mustTx(t, pool, func(st *Store) error {
		var err error
		view, err = st.PublishView(ctx, scope, id, claim, evaldomain.Comparison{Baseline: "a", Candidate: "b", Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}, true)
		return err
	})
	if view.Summary.Counts["a"].Expected != 5000 || view.Summary.TerminalPairs != 3750 || view.Summary.CompleteQualityPairs != 2500 || view.Summary.CompleteTokenPairs != 2500 {
		t.Fatalf("full denominator %+v", view.Summary)
	}
	counter := &evalQueryCounter{DBTX: pool}
	reader := NewPostgresStore(counter)
	p := SelectedPageParams{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100}
	seen := map[string]bool{}
	var maxBytes int
	for pageNumber := 0; pageNumber < 51; pageNumber++ {
		before := counter.queries
		page, err := reader.PairPage(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		if counter.queries-before != 2 || page.FilteredCount != 5000 || len(page.Items) > 100 {
			t.Fatal("unbounded page", counter.queries-before, page.FilteredCount, len(page.Items))
		}
		raw := bytesOf(page.Items)
		maxBytes = max(maxBytes, len(raw))
		if len(raw) > evaldomain.MaxDocumentBytes || strings.Contains(string(raw), "PRIVATE_") {
			t.Fatal("page bytes or private boundary", len(raw))
		}
		for _, pair := range page.Items {
			if seen[pair.ID] {
				t.Fatal("repeated pair")
			}
			seen[pair.ID] = true
		}
		if !page.HasMore {
			break
		}
		p.AfterOrdinal = page.LastOrdinal
	}
	if len(seen) != 5000 {
		t.Fatal("dropped pair", len(seen))
	}
	p.AfterOrdinal = -1
	p.SuiteID = "suite-00"
	page, err := reader.SelectedMemberPage(ctx, p)
	if err != nil || page.FilteredCount != 200 || len(page.Items) != 100 {
		t.Fatal("suite member page", page.FilteredCount, err)
	}
	cohort, err := reader.ChartCohort(ctx, scope.OwnerID, id, view.Generation, "", "tokens")
	if err != nil || cohort.Coverage.Included != 2500 || len(cohort.Bins) > evaldomain.MaxChartBins || *cohort.Distributions["a"].P50 != 45 || *cohort.Distributions["a"].P90 != 80 {
		t.Fatal("whole metric cohort", err)
	}
	p.OwnerID = "foreign"
	foreign, err := reader.PairPage(ctx, p)
	if err != nil || foreign.FilteredCount != 0 || len(foreign.Items) != 0 {
		t.Fatal("foreign pairs", err)
	}
	if _, err = reader.ChartCohort(ctx, "foreign", id, view.Generation, "", "tokens"); !code(err, "eval_not_found") {
		t.Fatal("foreign chart", err)
	}
	t.Logf("10,000 members: 50 pair pages; 2 SQL queries/page; maximum page %d bytes; %d metric bins", maxBytes, len(cohort.Bins))
}
