package evalservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// freezeInsertTracer counts client statements that insert member or plan
// resource rows, by table. The projection trigger's inserts run inside the
// server and are not counted.
type freezeInsertTracer struct {
	mu         sync.Mutex
	statements map[string]int
}

var freezeInsertStatement = regexp.MustCompile(`(?i)^\s*insert\s+into\s+(eval_members|eval_plan_resources)\s*\(`)

func (f *freezeInsertTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if match := freezeInsertStatement.FindStringSubmatch(data.SQL); match != nil {
		f.mu.Lock()
		f.statements[strings.ToLower(match[1])]++
		f.mu.Unlock()
	}
	return ctx
}
func (*freezeInsertTracer) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// freezeInserts runs fn with the service on a traced pool and returns the
// member and plan resource INSERT statements it sent, by table.
func (h *serviceHarness) freezeInserts(t *testing.T, fn func()) map[string]int {
	t.Helper()
	trace := &freezeInsertTracer{statements: map[string]int{}}
	config := h.pool.Config()
	config.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	original := h.service.pool
	h.service.pool = traced
	defer func() { h.service.pool = original }()
	fn()
	trace.mu.Lock()
	defer trace.mu.Unlock()
	return maps.Clone(trace.statements)
}

// assertFreezeStatements requires the exact number of member and plan
// resource INSERT statements: a freeze inserts each table in batches of at
// most 1,000 rows and 4 MiB of stored recipe or document bytes (LIM-EVM-18).
func assertFreezeStatements(t *testing.T, pool *pgxpool.Pool, id string, statements map[string]int, members, resources int) {
	t.Helper()
	var memberRows, resourceRows int
	if err := pool.QueryRow(t.Context(), `SELECT (SELECT count(*) FROM eval_members WHERE experiment_id = $1), (SELECT count(*) FROM eval_plan_resources WHERE experiment_id = $1)`, id).Scan(&memberRows, &resourceRows); err != nil {
		t.Fatal(err)
	}
	t.Logf("froze %d members and %d plan resources with %v INSERT statements", memberRows, resourceRows, statements)
	if statements["eval_members"] != members || statements["eval_plan_resources"] != resources {
		t.Errorf("want %d member and %d resource INSERT statements, got %v", members, resources, statements)
	}
}

// projectionQueue reads the experiment's projection queue row. During a
// freeze only the eval_members_projection statement trigger writes it: it
// creates the row or advances an existing row once per invocation.
func projectionQueue(t *testing.T, pool *pgxpool.Pool, id string) (revision, published int64, exists bool) {
	t.Helper()
	err := pool.QueryRow(t.Context(), `SELECT revision, published_revision FROM eval_projection_queue WHERE experiment_id = $1`, id).Scan(&revision, &published)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, 0, false
	}
	if err != nil {
		t.Fatal(err)
	}
	return revision, published, true
}

// assertMemberProjections requires exactly one projection row per frozen
// member, each created once (revision 1) by the freeze.
func assertMemberProjections(t *testing.T, pool *pgxpool.Pool, id string, members int) {
	t.Helper()
	var rows, created, matched int
	err := pool.QueryRow(t.Context(), `
SELECT count(*), count(*) FILTER (WHERE p.revision = 1), count(m.member_id)
  FROM eval_member_projections p
  LEFT JOIN eval_members m USING (experiment_id, member_id)
 WHERE p.experiment_id = $1`, id).Scan(&rows, &created, &matched)
	if err != nil {
		t.Fatal(err)
	}
	if rows != members || created != members || matched != members {
		t.Fatalf("member projections: %d rows, %d created once, %d matching members; want %d", rows, created, matched, members)
	}
}

type frozenMemberRow struct {
	ID, Pair, Suite, Case, Variant, Eligibility, Kind, Key, CaseSHA256, BindingSHA256 string
	Ordinal, Sample                                                                   int
	Reason                                                                            *string
	Recipe                                                                            []byte
}

// assertFrozenMembers compares every stored member with the row the frozen
// plan defines: manifest identity, execution-order ordinal, pair, the
// server-scoped submission key, execution kind, eligibility reason and the
// exact json.Marshal bytes of evalstore.Recipe. cases is keyed by case ID.
func assertFrozenMembers(t *testing.T, pool *pgxpool.Pool, e evalstore.Experiment, cases map[string]evaldomain.Case) {
	t.Helper()
	ctx := t.Context()
	plan, err := evalstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlan(ctx, e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	var manifest evaldomain.PublicPlan
	var order []string
	reasons := map[string]*string{}
	if plan.Document.Kind() == "ExternalRegistration" {
		var reg evaldomain.ExternalRegistration
		if err = json.Unmarshal(plan.Document.Bytes(), &reg); err != nil {
			t.Fatal(err)
		}
		manifest = reg.Manifest
		for _, m := range manifest.Members {
			order = append(order, m.MemberID)
		}
	} else {
		if manifest, err = evaldomain.PublicPlanProjection(plan.Document); err != nil {
			t.Fatal(err)
		}
		var native struct {
			Order   []string `json:"execution_order"`
			Members []struct {
				ID     string  `json:"member_id"`
				Reason *string `json:"reason"`
			} `json:"members"`
		}
		if err = json.Unmarshal(plan.Document.Bytes(), &native); err != nil {
			t.Fatal(err)
		}
		order = native.Order
		for _, m := range native.Members {
			reasons[m.ID] = m.Reason
		}
	}
	var setup struct {
		Variants []evaldomain.Variant `json:"variants"`
	}
	if err = json.Unmarshal(plan.Setup, &setup); err != nil {
		t.Fatal(err)
	}
	variants := map[string]evaldomain.Variant{}
	for _, v := range setup.Variants {
		variants[v.ID] = v
	}
	ordinals := map[string]int{}
	for i, id := range order {
		ordinals[id] = i
	}
	want := make([]frozenMemberRow, len(manifest.Members))
	for _, m := range manifest.Members {
		v := variants[m.VariantID]
		visible, err := evaldomain.ExecutionProjection(cases[m.CaseID])
		if err != nil {
			t.Fatal(err)
		}
		recipe, err := json.Marshal(evalstore.Recipe{Case: visible, Variant: v})
		if err != nil {
			t.Fatal(err)
		}
		pair, err := evaldomain.PairID(e.PortableID, m.SuiteID, m.CaseID, m.Sample)
		if err != nil {
			t.Fatal(err)
		}
		scoped, err := json.Marshal([]string{e.ID, m.MemberID})
		if err != nil {
			t.Fatal(err)
		}
		kind := "run"
		if v.Kind == "audit" {
			kind = "audit"
		}
		want[ordinals[m.MemberID]] = frozenMemberRow{
			ID: m.MemberID, Pair: pair, Suite: m.SuiteID, Case: m.CaseID, Variant: m.VariantID,
			Eligibility: m.Eligibility, Kind: kind, Key: "eval-" + evaldomain.Digest(scoped)[7:],
			CaseSHA256: m.CaseSHA256, BindingSHA256: m.BindingSHA256, Ordinal: ordinals[m.MemberID],
			Sample: m.Sample, Reason: reasons[m.MemberID], Recipe: recipe,
		}
	}
	rows, err := pool.Query(ctx, `
SELECT member_id, pair_id, suite_id, case_id, variant_id, eligibility, execution_kind,
       submission_key, case_sha256, binding_sha256, ordinal, sample, eligibility_reason, recipe
  FROM eval_members
 WHERE experiment_id = $1
 ORDER BY ordinal`, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	got, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (frozenMemberRow, error) {
		var r frozenMemberRow
		err := row.Scan(&r.ID, &r.Pair, &r.Suite, &r.Case, &r.Variant, &r.Eligibility, &r.Kind,
			&r.Key, &r.CaseSHA256, &r.BindingSHA256, &r.Ordinal, &r.Sample, &r.Reason, &r.Recipe)
		return r, err
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("stored %d members, plan defines %d", len(got), len(want))
	}
	for i := range want {
		g, w := got[i], want[i]
		if !bytes.Equal(g.Recipe, w.Recipe) {
			t.Fatalf("member %d recipe bytes differ:\n got %s\nwant %s", i, g.Recipe, w.Recipe)
		}
		if !reflect.DeepEqual(g, w) {
			t.Fatalf("member %d differs:\n got %+v\nwant %+v", i, g, w)
		}
	}
}

// externalAtByteCeiling builds the largest external registration whose
// CreateExperiment document fits MaxDocumentBytes. Each case has one sample
// for both variants and shares one input artifact, so byte admission, not
// the 10,000-member format ceiling, bounds the member count.
func (h *serviceHarness) externalAtByteCeiling(t *testing.T) (evaldomain.Frozen, map[string]evaldomain.Case, int) {
	t.Helper()
	const portable = "external-byte-ceiling"
	draft, data := h.dataset(t, "workflow")
	pre := map[string]Preflight{}
	for _, v := range draft.Variants {
		p, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, v, data.Cases)
		if err != nil {
			t.Fatal(err)
		}
		pre[v.ID] = p
	}
	bundle, err := BuildPlan(portable, time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC), draft, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := evaldomain.PublicPlanProjection(bundle.Plan)
	if err != nil {
		t.Fatal(err)
	}
	var setup struct {
		Checks []evaldomain.Check `json:"checks"`
	}
	if err = json.Unmarshal(bundle.Setup, &setup); err != nil {
		t.Fatal(err)
	}
	bindings := map[string]string{}
	for _, m := range manifest.Members {
		bindings[m.VariantID] = m.BindingSHA256
	}
	suite, template := manifest.Members[0].SuiteID, data.Cases[0]
	budgets := draft.Budgets
	budgets.MaxMembers = evaldomain.MaxMembers
	build := func(count int) (evaldomain.CreateExperiment, map[string]evaldomain.Case) {
		members := make([]evaldomain.PublicMember, 0, 2*count)
		recipes := make([]evaldomain.MemberRecipe, 0, 2*count)
		cases := make(map[string]evaldomain.Case, count)
		for i := range count {
			c := template
			c.ID = fmt.Sprintf("ceiling-%04d", i)
			encoded, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			cases[c.ID] = c
			for _, v := range draft.Variants {
				id, err := evaldomain.MemberID(portable, suite, c.ID, 1, v.ID)
				if err != nil {
					t.Fatal(err)
				}
				members = append(members, evaldomain.PublicMember{MemberID: id, SuiteID: suite, CaseID: c.ID, VariantID: v.ID, Sample: 1, CaseSHA256: evaldomain.Digest(encoded), BindingSHA256: bindings[v.ID], Eligibility: "eligible"})
				recipes = append(recipes, evaldomain.MemberRecipe{MemberID: id, Case: c})
			}
		}
		plan := manifest
		plan.Members = members
		return evaldomain.CreateExperiment{Name: "External byte ceiling", ControlMode: "external", Registration: &evaldomain.ExternalRegistration{
			SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: bundle.Plan.Digest(), Manifest: plan,
			Source: evaldomain.Source{System: "fixture", ID: "byte-ceiling"}, Variants: draft.Variants, Recipes: recipes,
			Checks: setup.Checks, Comparison: draft.Comparison, Budgets: budgets,
		}}, cases
	}
	size := func(count int) int {
		input, _ := build(count)
		encoded, err := json.Marshal(input)
		if err != nil {
			t.Fatal(err)
		}
		return len(encoded)
	}
	low, high := 1, evaldomain.MaxMembers/2
	if size(high) <= evaldomain.MaxDocumentBytes {
		t.Fatal("the member format ceiling, not the document bound, limits this registration")
	}
	for low < high {
		if mid := (low + high + 1) / 2; size(mid) <= evaldomain.MaxDocumentBytes {
			low = mid
		} else {
			high = mid - 1
		}
	}
	input, cases := build(low)
	doc := frozen(t, "CreateExperiment", input)
	if len(doc.Bytes()) > evaldomain.MaxDocumentBytes || size(low+1) <= evaldomain.MaxDocumentBytes {
		t.Fatalf("registration of %d cases is not at the byte ceiling", low)
	}
	t.Logf("external registration at the byte ceiling: %d of %d bytes, %d members (format ceiling %d)", len(doc.Bytes()), evaldomain.MaxDocumentBytes, 2*low, evaldomain.MaxMembers)
	return doc, cases, 2 * low
}

// The largest external registration that byte admission accepts freezes its
// 1,216 members with two bounded INSERT statements. The statement trigger
// runs once per statement: the first creates the projection queue row at
// revision 1 and the second advances it, with one projection row per member.
// The per-member loop sent one INSERT per member and left the queue at a
// revision equal to the member count.
func TestSharedPostgresExternalByteCeilingFreezesMembersInBoundedStatements(t *testing.T) {
	h := newHarness(t)
	doc, cases, members := h.externalAtByteCeiling(t)
	var receipt evalstore.Receipt
	statements := h.freezeInserts(t, func() {
		var err error
		receipt, err = h.service.Create(t.Context(), h.scope, doc, identity(t, "external-byte-ceiling", 0, doc))
		if err != nil {
			t.Fatal(err)
		}
	})
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	if e.State != evaldomain.StateReady || e.Expected != members {
		t.Fatalf("registration: state=%s members=%d, want ready with %d", e.State, e.Expected, members)
	}
	// 1,216 small recipes: two member batches (1,000 + 216 rows), and the two
	// binding resources in one batch.
	assertFreezeStatements(t, h.pool, e.ID, statements, 2, 1)
	if revision, published, exists := projectionQueue(t, h.pool, e.ID); !exists || revision != 2 || published != 0 {
		t.Errorf("projection queue row: exists=%v revision=%d published=%d, want created by the first batch and advanced by the second", exists, revision, published)
	}
	assertMemberProjections(t, h.pool, e.ID, members)
	assertFrozenMembers(t, h.pool, e, cases)
}
