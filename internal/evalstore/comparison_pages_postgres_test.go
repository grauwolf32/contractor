package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
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

// seedSelectedView seeds external experiment id of n members directly at the
// persistence boundary, independently of portable plan authoring size: pairs
// of the pair-page trace, pairsPerSuite pairs per suite, every member's
// projection selected. vary, when set, changes each member's view and
// collection completeness before the view is published through PublishView.
func seedSelectedView(t *testing.T, pool *pgxpool.Pool, scope Scope, id string, n, pairsPerSuite int, vary func(ordinal int, view *evaldomain.MemberView, complete *bool)) *View {
	t.Helper()
	ctx := t.Context()
	template := createExperiment(t, pool, scope, id+"-template", "trace-1", "external-workflow")
	var trace struct {
		Items []evaldomain.Pair `json:"items"`
	}
	raw := contractstest.ReadFile(t, "api", "testdata", "evals", "valid", "pair-page.json")
	if err := json.Unmarshal(raw, &trace); err != nil {
		t.Fatal(err)
	}
	memberRows := make([][]any, 0, n)
	projectionRows := make([][]any, 0, n)
	for ordinal := 0; ordinal < n; ordinal++ {
		pairNumber := ordinal / 2
		pair := trace.Items[pairNumber%len(trace.Items)]
		v := pair.A
		if ordinal%2 == 1 {
			v = pair.B
		}
		// Avoid sharing fixture pointer fields across samples.
		var cloned evaldomain.MemberView
		if err := json.Unmarshal(bytesOf(v), &cloned); err != nil {
			t.Fatal(err)
		}
		v = cloned
		suite := fmt.Sprintf("suite-%02d", pairNumber/pairsPerSuite)
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
		if vary != nil {
			vary(ordinal, &v, &complete)
		}
		memberRows = append(memberRows, []any{id, memberID, pairID, ordinal, suite, caseID, 1, v.Member.VariantID, v.Member.CaseSHA256, v.Member.BindingSHA256, v.Member.Eligibility, "run", []byte("{}"), fmt.Sprintf("%s-%d", id, ordinal)})
		projectionRows = append(projectionRows, []any{memberID, bytesOf(v), complete})
	}
	mustTx(t, pool, func(st *Store) error {
		// Bulk fixture setup runs without the test pool's statement timeout;
		// measured page queries keep it.
		if _, err := st.db.Exec(ctx, `SET LOCAL statement_timeout = 0`); err != nil {
			return err
		}
		_, err := st.db.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms) VALUES($1,$2,$3,$1,'external','Large selected fixture','finished',$4,1,1000)`, id, scope.OwnerID, scope.ProjectID, n)
		if err != nil {
			return err
		}
		_, err = st.db.Exec(ctx, `UPDATE eval_experiments e SET plan_kind=p.plan_kind,plan_document=p.plan_document,plan_sha256=p.plan_sha256,plan_setup=p.plan_setup,plan_readiness=p.plan_readiness,plan_created_at=clock_timestamp(),updated_at=clock_timestamp() FROM eval_experiments p WHERE e.experiment_id=$1 AND p.experiment_id=$2`, id, template.ID)
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
	claims, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Claim(ctx, "reader-"+id, 5*time.Minute, 10)
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
		t.Fatal("selected fixture was not claimable")
	}
	var view *View
	mustTx(t, pool, func(st *Store) error {
		var err error
		view, err = st.PublishView(ctx, scope, id, claim, evaldomain.Comparison{Baseline: "a", Candidate: "b", Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}, true)
		return err
	})
	return view
}

// requireProjectedSides checks that every pair row of experiment id stores,
// for each side, exactly the member fields its own document holds and the
// side member's frozen ordinal (EVM-R245).
func requireProjectedSides(t *testing.T, db pg.DBTX, id string) int {
	t.Helper()
	ctx := t.Context()
	rows, err := db.Query(ctx, `SELECT p.document, p.member_ordinal_a, p.member_ordinal_b, p.variant_a, p.variant_b,
    p.eligibility_a, p.eligibility_b, p.execution_state_a, p.execution_state_b, p.assessment_a, p.assessment_b,
    p.conflicting_a, p.conflicting_b, a.ordinal, b.ordinal, a.member_id, b.member_id
FROM eval_view_pairs p
JOIN eval_members a ON a.experiment_id = p.experiment_id AND a.ordinal = p.member_ordinal_a
JOIN eval_members b ON b.experiment_id = p.experiment_id AND b.ordinal = p.member_ordinal_b
WHERE p.experiment_id = $1
ORDER BY p.generation, p.ordinal`, id)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	checked := 0
	for rows.Next() {
		var document []byte
		var ordinals, frozen [2]int
		var variants, eligibility, assessments, members [2]string
		var states [2]*string
		var conflicting [2]bool
		if err = rows.Scan(&document, &ordinals[0], &ordinals[1], &variants[0], &variants[1], &eligibility[0], &eligibility[1],
			&states[0], &states[1], &assessments[0], &assessments[1], &conflicting[0], &conflicting[1],
			&frozen[0], &frozen[1], &members[0], &members[1]); err != nil {
			t.Fatal(err)
		}
		var pair evaldomain.Pair
		if err = json.Unmarshal(document, &pair); err != nil {
			t.Fatal(err)
		}
		for i, side := range []evaldomain.MemberView{pair.A, pair.B} {
			if members[i] != side.Member.ID || ordinals[i] != frozen[i] || variants[i] != side.Member.VariantID ||
				eligibility[i] != side.Member.Eligibility || assessments[i] != side.Assessment || conflicting[i] != side.Conflicting ||
				!reflect.DeepEqual(states[i], executionState(side)) {
				t.Fatalf("pair %s side %d stores %q %d %q %q %v %q %t, its document holds %+v", pair.ID, i, members[i], ordinals[i],
					variants[i], eligibility[i], states[i], assessments[i], conflicting[i], side)
			}
		}
		checked++
	}
	if err = rows.Err(); err != nil {
		t.Fatal(err)
	}
	return checked
}

// referencePairCohortSQL and referenceMemberCohortSQL are the page statements
// as they were before V409-014, kept verbatim with referencePage as the
// reference the single-statement pages must reproduce: a count statement and
// a page statement over a cohort that carries every document, the member
// cohort decoding every pair document as jsonb.
var referencePairCohortSQL = `
WITH filtered AS MATERIALIZED (
    SELECT p.ordinal, p.document, abs(%s - %s) AS difference
    FROM eval_view_pairs p
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND ($5 IN ('', 'all')
            OR ($5 = 'regressions' AND p.regression)
            OR ($5 = 'unresolved' AND p.unresolved))
        AND (NOT $6 OR (
            (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
            OR (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
        ))
        AND ($10 = '' OR (%s IS NOT NULL AND %s IS NOT NULL))
)
`

var referenceMemberCohortSQL = `
WITH members AS (
    SELECT member.ordinal, convert_to(side.value::text, 'UTF8') AS document,
        side.collection_complete, side.value,
        CASE side.arm WHEN 'a' THEN %s ELSE %s END AS measure
    FROM eval_view_pairs p
    JOIN eval_experiments e USING (experiment_id)
    CROSS JOIN LATERAL (VALUES
        ('a', convert_from(p.document, 'UTF8')::jsonb -> 'a', p.complete_a),
        ('b', convert_from(p.document, 'UTF8')::jsonb -> 'b', p.complete_b)
    ) AS side(arm, value, collection_complete)
    JOIN eval_members member ON member.experiment_id = p.experiment_id
        AND member.member_id = side.value #>> '{member,memberId}'
    WHERE e.owner_id = $1 AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND (%s IS NOT NULL AND %s IS NOT NULL OR NOT $7)
), filtered AS MATERIALIZED (
    SELECT ordinal, document, NULL::double precision AS difference
    FROM members
    WHERE ($5 = '' OR value #>> '{member,variantId}' = $5)
        AND ($6 IN ('', 'all')
            OR ($6 = 'unresolved' AND (
                value #>> '{execution,state}' NOT IN ('succeeded', 'failed', 'cancelled')
                OR value ->> 'assessment' NOT IN ('pass', 'fail')
                OR NOT collection_complete OR (value ->> 'conflicting')::boolean
            ))
            OR ($6 = 'failed' AND value #>> '{execution,state}' = 'failed')
            OR ($6 = 'unscored' AND value ->> 'assessment' = 'unscored')
            OR ($6 IN ('unsupported', 'blocked') AND value #>> '{member,eligibility}' = $6)
            OR ($6 = 'conflicting' AND (value ->> 'conflicting')::boolean)
        )
        AND (NOT $7 OR (measure >= $8 AND (measure < $9 OR ($10 AND measure = $9))))
)
`

func referencePairPage(ctx context.Context, db pg.DBTX, p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) {
	a, b := pairMeasureColumns(p.Metric)
	if p.Bin != nil {
		a, b = pairMeasureColumns(p.Bin.Metric)
	}
	bin, lo, hi, inclusive := p.binArgs()
	query := fmt.Sprintf(referencePairCohortSQL, b, a, a, a, a, b, b, b, a, b)
	return referencePage[evaldomain.Pair](ctx, db, query, []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.Filter, bin, lo, hi, inclusive, p.Metric}, p)
}

func referenceMemberPage(ctx context.Context, db pg.DBTX, p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
	metric := "tokens"
	if p.Bin != nil {
		metric = p.Bin.Metric
	}
	a, b := pairMeasureColumns(metric)
	bin, lo, hi, inclusive := p.binArgs()
	query := fmt.Sprintf(referenceMemberCohortSQL, a, b, a, b)
	return referencePage[evaldomain.MemberView](ctx, db, query, []any{p.OwnerID, p.ExperimentID, p.Generation, p.SuiteID, p.VariantID, p.Filter, bin, lo, hi, inclusive}, p)
}

func referencePage[T any](ctx context.Context, db pg.DBTX, query string, args []any, p SelectedPageParams) (SelectedPage[T], error) {
	out := SelectedPage[T]{Items: []T{}, LastOrdinal: -1}
	if err := db.QueryRow(ctx, query+"SELECT count(*) FROM filtered", args...).Scan(&out.FilteredCount); err != nil {
		return out, err
	}
	order, after := "ordinal", "ordinal>$11"
	args = append(args, p.AfterOrdinal)
	if p.Absolute {
		order = "difference DESC, ordinal"
		after = "($11::integer<0 OR difference<$12 OR (difference=$12 AND ordinal>$11))"
		args = append(args, p.AfterDifference)
	}
	args = append(args, p.Limit+1)
	pageSQL := fmt.Sprintf("SELECT ordinal,document,difference FROM filtered WHERE %s ORDER BY %s LIMIT $%d", after, order, len(args))
	rows, err := db.Query(ctx, query+pageSQL, args...)
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var ordinal int
		var raw []byte
		var difference *float64
		if err = rows.Scan(&ordinal, &raw, &difference); err != nil {
			return out, err
		}
		if len(out.Items) == p.Limit {
			out.HasMore = true
			break
		}
		var item T
		if err = json.Unmarshal(raw, &item); err != nil {
			return out, err
		}
		out.Items = append(out.Items, item)
		out.LastOrdinal = ordinal
		out.LastDifference = difference
	}
	return out, rows.Err()
}

// requireSamePages walks every page of p with read and with the reference and
// requires identical pages, then the empty page after the last cursor. It
// returns the pages read.
func requireSamePages[T any](t *testing.T, name string, p SelectedPageParams, read, reference func(SelectedPageParams) (SelectedPage[T], error)) int {
	t.Helper()
	pages := 0
	for {
		got, err := read(p)
		if err != nil {
			t.Fatalf("%s after %d: %v", name, p.AfterOrdinal, err)
		}
		want, err := reference(p)
		if err != nil {
			t.Fatalf("%s reference after %d: %v", name, p.AfterOrdinal, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s after %d differs from the two-statement reference:\n got %+v\nwant %+v", name, p.AfterOrdinal, got, want)
		}
		pages++
		if len(got.Items) == 0 {
			return pages
		}
		// Past the last page this asks for the empty page after the cursor.
		p.AfterOrdinal, p.AfterDifference = got.LastOrdinal, got.LastDifference
	}
}

// Every supported pair and member filter, sort and cursor, including empty,
// end and foreign-owner pages, reads in one statement exactly what the
// two-statement reference read: the filtered total, the complete items in
// order and the continuation (EVM-R245). The fixture varies every projected
// member field, NULL executions and measures, ties and bin boundaries.
func TestSharedPostgresEvalSelectedPagesMatchTheTwoStatementReference(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "evaluation")
	const id = "varied-comparison"
	view := seedSelectedView(t, pool, scope, id, 80, 15, func(ordinal int, v *evaldomain.MemberView, complete *bool) {
		switch ordinal % 7 {
		case 3:
			v.Execution = nil
		case 4:
			if v.Execution != nil {
				v.Execution.State = "running"
			}
		case 5:
			if v.Execution != nil {
				v.Execution.State = "cancelled"
			}
		}
		if ordinal%9 < 5 {
			v.Assessment = []string{"pass", "fail", "unscored", "incomplete", "error"}[ordinal%9]
		}
		switch ordinal % 11 {
		case 0:
			v.Member.Eligibility = "unsupported"
		case 1:
			v.Member.Eligibility = "blocked"
		}
		v.Conflicting = ordinal%13 == 2
		*complete = ordinal%3 != 0
	})
	if checked := requireProjectedSides(t, pool, id); checked != 40 {
		t.Fatalf("checked %d pair rows, want 40", checked)
	}
	bins := []*BinFilter{nil}
	for _, metric := range []string{"tokens", "duration"} {
		a, b := pairMeasureColumns(metric)
		var low, middle float64
		if err := pool.QueryRow(ctx, fmt.Sprintf(`SELECT min(v), percentile_disc(0.5) WITHIN GROUP (ORDER BY v)
FROM (SELECT %s AS v FROM eval_view_pairs p WHERE experiment_id = $1 UNION ALL SELECT %s FROM eval_view_pairs p WHERE experiment_id = $1) measures`, a, b), id).Scan(&low, &middle); err != nil {
			t.Fatal(err)
		}
		// Both bounds are measures of the fixture, so a measure equals the
		// upper bound; when every measure is equal, the exclusive bin is empty.
		bins = append(bins, &BinFilter{Metric: metric, Lower: low, Upper: middle}, &BinFilter{Metric: metric, Lower: low, Upper: middle, UpperInclusive: true})
	}
	reader := NewPostgresStore(pool, artifactpgtest.NewStorage())
	pages := 0
	base := SelectedPageParams{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1}
	for _, suite := range []string{"", "suite-00", "suite-02", "suite-none"} {
		for _, filter := range []string{"", "all", "regressions", "unresolved"} {
			for _, bin := range bins {
				for _, metric := range []string{"", "tokens", "duration"} {
					for _, absolute := range []bool{false, true} {
						p := base
						p.SuiteID, p.Filter, p.Bin, p.Metric, p.Absolute, p.Limit = suite, filter, bin, metric, absolute, 3
						name := fmt.Sprintf("pairs suite=%q filter=%q bin=%+v metric=%q absolute=%t", suite, filter, bin, metric, absolute)
						pages += requireSamePages(t, name, p,
							func(p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) { return reader.PairPage(ctx, p) },
							func(p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) {
								return referencePairPage(ctx, pool, p)
							})
					}
				}
			}
		}
	}
	for _, suite := range []string{"", "suite-01", "suite-none"} {
		for _, variant := range []string{"", "a", "b"} {
			for _, filter := range []string{"", "all", "unresolved", "failed", "unscored", "unsupported", "blocked", "conflicting"} {
				for _, bin := range bins {
					p := base
					p.SuiteID, p.VariantID, p.Filter, p.Bin, p.Limit = suite, variant, filter, bin, 7
					name := fmt.Sprintf("members suite=%q variant=%q filter=%q bin=%+v", suite, variant, filter, bin)
					pages += requireSamePages(t, name, p,
						func(p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
							return reader.SelectedMemberPage(ctx, p)
						},
						func(p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
							return referenceMemberPage(ctx, pool, p)
						})
				}
			}
		}
	}
	// One whole page, a cursor beyond the last ordinal, a foreign owner and a
	// superseded generation.
	for _, p := range []SelectedPageParams{
		{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100},
		{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: 1000, Limit: 100},
		{OwnerID: "foreign", ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100},
		{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation + 1, AfterOrdinal: -1, Limit: 100},
	} {
		pages += requireSamePages(t, fmt.Sprintf("pairs %+v", p), p,
			func(p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) { return reader.PairPage(ctx, p) },
			func(p SelectedPageParams) (SelectedPage[evaldomain.Pair], error) {
				return referencePairPage(ctx, pool, p)
			})
		pages += requireSamePages(t, fmt.Sprintf("members %+v", p), p,
			func(p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
				return reader.SelectedMemberPage(ctx, p)
			},
			func(p SelectedPageParams) (SelectedPage[evaldomain.MemberView], error) {
				return referenceMemberPage(ctx, pool, p)
			})
	}
	whole, err := reader.SelectedMemberPage(ctx, SelectedPageParams{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100})
	if err != nil || whole.FilteredCount != 80 || len(whole.Items) != 80 || whole.HasMore {
		t.Fatalf("whole member page: %d of %d, more %t, %v", len(whole.Items), whole.FilteredCount, whole.HasMore, err)
	}
	foreign, err := reader.PairPage(ctx, SelectedPageParams{OwnerID: "foreign", ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100})
	if err != nil || foreign.FilteredCount != 0 || len(foreign.Items) != 0 || foreign.HasMore || foreign.LastOrdinal != -1 {
		t.Fatalf("foreign pairs %+v: %v", foreign, err)
	}
	t.Logf("compared %d pages of 40 pairs and 80 members with the two-statement reference", pages)
}

// A page reads the documents of its own pairs only: a member page whose
// cohort holds a damaged pair document outside the page still reads, with the
// exact count, while the reference decoded every document of the cohort and
// failed. The page that holds the damaged pair fails. A document that keeps a
// NUL escape in a free string, which PostgreSQL cannot decode as JSON, reads
// on both pages.
func TestSharedPostgresEvalSelectedPagesDecodeOnlyTheirOwnDocuments(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "evaluation")
	const id = "damaged-comparison"
	reason := "reason with a NUL \x00 and a \\u0000 text"
	view := seedSelectedView(t, pool, scope, id, 20, 100, func(ordinal int, v *evaldomain.MemberView, _ *bool) {
		if ordinal == 0 {
			v.Member.Reason = &reason
		}
	})
	if checked := requireProjectedSides(t, pool, id); checked != 10 {
		t.Fatalf("checked %d pair rows, want 10", checked)
	}
	reader := NewPostgresStore(pool, artifactpgtest.NewStorage())
	first := SelectedPageParams{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 4}
	members, err := reader.SelectedMemberPage(ctx, first)
	if err != nil || len(members.Items) != 4 || members.Items[0].Member.Reason == nil || *members.Items[0].Member.Reason != reason {
		t.Fatalf("member page with a NUL escape: %+v, %v", members.Items, err)
	}
	pairs, err := reader.PairPage(ctx, first)
	if err != nil || len(pairs.Items) != 4 || pairs.Items[0].A.Member.Reason == nil || *pairs.Items[0].A.Member.Reason != reason {
		t.Fatalf("pair page with a NUL escape: %+v, %v", pairs.Items, err)
	}
	var pgErr *pgconn.PgError
	if _, err = referenceMemberPage(ctx, pool, first); pg.Classify(err) != pg.ClassInvalid || !errors.As(err, &pgErr) || !strings.Contains(pgErr.Message, "Unicode escape") {
		t.Fatalf("the reference member page decoded the NUL escape as jsonb: %v", err)
	}
	// Only DDL can bypass eval_view_pairs_immutable.
	for _, statement := range []string{
		`ALTER TABLE eval_view_pairs DISABLE TRIGGER eval_view_pairs_immutable`,
		`UPDATE eval_view_pairs SET document = '\xff'::bytea WHERE experiment_id = 'damaged-comparison' AND ordinal = 9`,
		`ALTER TABLE eval_view_pairs ENABLE TRIGGER eval_view_pairs_immutable`,
	} {
		if _, err = pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	all := first
	all.Filter = "all"
	members, err = reader.SelectedMemberPage(ctx, all)
	if err != nil || members.FilteredCount != 20 || len(members.Items) != 4 || !members.HasMore {
		t.Fatalf("member page beside a damaged pair: %d of %d, %v", len(members.Items), members.FilteredCount, err)
	}
	pairs, err = reader.PairPage(ctx, all)
	if err != nil || pairs.FilteredCount != 10 || len(pairs.Items) != 4 || !pairs.HasMore {
		t.Fatalf("pair page beside a damaged pair: %d of %d, %v", len(pairs.Items), pairs.FilteredCount, err)
	}
	if _, err = referenceMemberPage(ctx, pool, all); err == nil {
		t.Fatal("the reference member page did not decode the damaged document outside its page")
	}
	last := all
	last.AfterOrdinal = 15
	if _, err = reader.SelectedMemberPage(ctx, last); err == nil {
		t.Fatal("the member page of the damaged pair decoded it")
	}
}

func TestSharedPostgresEvalTenThousandSelectedMembersBoundedPages(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "evaluation")
	const id = "large-comparison"
	view := seedSelectedView(t, pool, scope, id, evaldomain.MaxMembers, 100, nil)
	if view.Summary.Counts["a"].Expected != 5000 || view.Summary.TerminalPairs != 3750 || view.Summary.CompleteQualityPairs != 2500 || view.Summary.CompleteTokenPairs != 2500 {
		t.Fatalf("full denominator %+v", view.Summary)
	}
	counter := &evalQueryCounter{DBTX: pool}
	reader := NewPostgresStore(counter, artifactpgtest.NewStorage())
	p := SelectedPageParams{OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100}
	seen := map[string]bool{}
	var maxBytes int
	for pageNumber := 0; pageNumber < 51; pageNumber++ {
		before := counter.queries
		page, err := reader.PairPage(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		if counter.queries-before != 1 || page.FilteredCount != 5000 || len(page.Items) > 100 {
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
		p.AfterOrdinal = page.LastOrdinal
		if !page.HasMore {
			break
		}
	}
	if len(seen) != 5000 {
		t.Fatal("dropped pair", len(seen))
	}
	empty, err := reader.PairPage(ctx, p)
	if err != nil || empty.FilteredCount != 5000 || len(empty.Items) != 0 || empty.HasMore || empty.LastOrdinal != -1 {
		t.Fatalf("empty page after the last cursor: %+v, %v", empty, err)
	}
	p.AfterOrdinal = -1
	p.SuiteID = "suite-00"
	before := counter.queries
	page, err := reader.SelectedMemberPage(ctx, p)
	if err != nil || counter.queries-before != 1 || page.FilteredCount != 200 || len(page.Items) != 100 {
		t.Fatal("suite member page", counter.queries-before, page.FilteredCount, err)
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

	// The statement reads the cohort of 5,000 pair rows once and, for the
	// documents, at most one row per page item, in the custom and the generic
	// plan of a pair page, of a middle pair page sorted by difference and of
	// member pages. A custom plan's planning may probe an index's extremes,
	// so the budget allows explaintest's few rows beyond the execution's own.
	if _, err = pool.Exec(ctx, `ANALYZE eval_view_pairs, eval_experiments`); err != nil {
		t.Fatal(err)
	}
	middle := 40.0
	statements := map[string]SelectedPageParams{
		"pairs":           {OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Limit: 100},
		"pairs by delta":  {OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: 2500, AfterDifference: &middle, Metric: "tokens", Absolute: true, Limit: 100},
		"members":         {OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: 4999, Limit: 100},
		"members, failed": {OwnerID: scope.OwnerID, ExperimentID: id, Generation: view.Generation, AfterOrdinal: -1, Filter: "failed", VariantID: "a", Limit: 100},
	}
	for name, params := range statements {
		query, args := pairPageStatement(params)
		if strings.HasPrefix(name, "members") {
			query, args = memberPageStatement(params)
		}
		for _, mode := range explaintest.Modes {
			analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, query, args...)
			if err != nil {
				t.Fatal(name, mode, err)
			}
			if reads := analysis.Reads.Of("eval_view_pairs"); reads > 5000+int64(params.Limit)+8 {
				t.Errorf("%s page in a %s plan read %d pair rows, more than the cohort of 5000 and one per item:\n%+v",
					name, mode, reads, analysis.Plan)
			}
			t.Logf("%s page, 10,000 members in 5,000 pairs, %s plan: %d pair rows read\n%+v",
				name, mode, analysis.Reads.Of("eval_view_pairs"), analysis.Plan)
		}
	}
	t.Logf("10,000 members: 50 pair pages; 1 SQL statement/page; maximum page %d bytes; %d metric bins", maxBytes, len(cohort.Bins))
}
