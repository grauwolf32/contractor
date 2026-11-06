package evalservice

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// planReads runs fn with the service on a traced pool and returns how many
// eval_experiments reads it sent that select the plan document and how many
// that select only metadata.
func (h *serviceHarness) planReads(t *testing.T, fn func()) (full, metadata int) {
	t.Helper()
	trace := &planTransferTracer{}
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
	return trace.full, trace.metadata
}

// storedPlan reads experiment id's plan document, setup and stored readiness
// directly, and derives the readiness the experiment view computed from the
// document before the freeze stored it.
func (h *serviceHarness) storedPlan(t *testing.T, id string) (document, setup, readiness []byte, derived evaldomain.Readiness) {
	t.Helper()
	var kind, digest string
	if err := h.pool.QueryRow(t.Context(), `SELECT plan_kind, plan_document, plan_document_sha256, plan_setup, plan_readiness FROM eval_experiments WHERE experiment_id=$1`, id).
		Scan(&kind, &document, &digest, &setup, &readiness); err != nil {
		t.Fatal(err)
	}
	var prepared preparedSetup
	if err := json.Unmarshal(setup, &prepared); err != nil {
		t.Fatal(err)
	}
	derived, err := evaldomain.NativeReadiness(evaldomain.Restore(kind, document, digest), prepared.Comparison)
	if err != nil {
		t.Fatal(err)
	}
	return document, setup, readiness, derived
}

// At the native member ceiling, Get and the command view read one metadata
// projection of the frozen plan and no document, and Get returns the readiness
// the freeze stored: exactly what the plan document yields (EVM-R242). The
// stored readiness holds two arm counts, not one entry per member, so its
// size depends on the pinned dimensions only (LIM-EVM-19).
func TestSharedPostgresExperimentViewReadsStoredReadinessAtNativeCeiling(t *testing.T) {
	h := newHarness(t)
	e := h.preparedAtNativeCeiling(t, 1)
	document, setup, stored, derived := h.storedPlan(t, e.ID)
	var view ExperimentView
	full, metadata := h.planReads(t, func() {
		var err error
		view, err = h.service.Get(t.Context(), e.OwnerID, e.ID)
		if err != nil {
			t.Fatal(err)
		}
	})
	if full != 0 || metadata != 1 {
		t.Fatalf("Get read %d plan documents and %d metadata projections", full, metadata)
	}
	if view.Readiness == nil || !reflect.DeepEqual(*view.Readiness, derived) {
		t.Fatalf("Get readiness differs from the plan document's:\n got %+v\nwant %+v", view.Readiness, derived)
	}
	half := evaldomain.MaxNativeMembers / 2
	for i, arm := range view.Readiness.Arms {
		if arm.Expected != half || arm.Eligible != half || arm.VariantID != []string{"a", "b"}[i] {
			t.Fatalf("arm %d counts %+v at the member ceiling", i, arm)
		}
	}
	encoded, err := json.Marshal(derived)
	if err != nil {
		t.Fatal(err)
	}
	if string(stored) != string(encoded) || len(derived.Pins) > 2048 || view.PlanSHA256 == nil || string(view.Setup) != string(setup) {
		t.Fatalf("stored readiness %s, setup %s, plan %v", stored, view.Setup, view.PlanSHA256)
	}
	t.Logf("%d members: Get transfers %d setup and %d readiness bytes instead of a %d byte plan document; %d readiness pins",
		e.Expected, len(setup), len(stored), len(document), len(derived.Pins))

	var command string
	if err = h.pool.QueryRow(t.Context(), `SELECT command_id FROM eval_commands WHERE experiment_id=$1 AND kind='prepare'`, e.ID).Scan(&command); err != nil {
		t.Fatal(err)
	}
	full, metadata = h.planReads(t, func() {
		_, plan, err := h.service.GetCommand(t.Context(), e.OwnerID, e.ID, command)
		if err != nil || plan == nil || *plan != *view.PlanSHA256 {
			t.Fatalf("command plan %v: %v", plan, err)
		}
	})
	if full != 0 || metadata != 1 {
		t.Fatalf("GetCommand read %d plan documents and %d metadata projections", full, metadata)
	}
}

// drained runs an eight-member native workflow experiment to finished, so
// every member has a collected result in the published view.
func (h *serviceHarness) drained(t *testing.T) evalstore.Experiment {
	t.Helper()
	e := h.preparedWithCapacity(t, "workflow", 8)
	h.command(t, e, "start")
	c := h.coordinator(t, "drain")
	for range 32 {
		tick(t, c)
		h.finishRuns(t)
		tick(t, c)
		if e = h.get(t, e.ID); e.State == evaldomain.StateFinished {
			return e
		}
	}
	t.Fatalf("experiment did not drain: %s", e.State)
	return e
}

// The selected views, the report and the review context need only the plan
// identity and setup, so none of them transfers the plan document.
func TestSharedPostgresSelectedViewsReadPlanMetadataOnly(t *testing.T) {
	h := newHarness(t)
	e := h.drained(t)
	page := MemberPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, AfterOrdinal: -1, Limit: 10}
	pairs, err := h.service.Pairs(t.Context(), PairPageParams{MemberPageParams: page})
	if err != nil || len(pairs.Items) == 0 || pairs.Items[0].A.ResultSHA256 == nil {
		t.Fatalf("pairs %+v: %v", pairs.Items, err)
	}
	pair := pairs.Items[0]
	views := map[string]func() error{
		"get": func() error { _, err := h.service.Get(t.Context(), e.OwnerID, e.ID); return err },
		"members": func() error {
			_, err := h.service.Members(t.Context(), page)
			return err
		},
		"pairs": func() error {
			_, err := h.service.Pairs(t.Context(), PairPageParams{MemberPageParams: page})
			return err
		},
		"pair": func() error { _, err := h.service.Pair(t.Context(), page, pair.ID); return err },
		"chart": func() error {
			_, err := h.service.Chart(t.Context(), ChartParams{PairPageParams: PairPageParams{MemberPageParams: MemberPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, AfterOrdinal: -1}}, Chart: "quality"})
			return err
		},
		"report": func() error { _, err := h.service.Report(t.Context(), page); return err },
		"review": func() error {
			_, err := h.service.Review(t.Context(), e.OwnerID, e.ID, pair.A.Member.ID, "")
			return err
		},
	}
	for name, view := range views {
		full, metadata := h.planReads(t, func() {
			if err := view(); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
		})
		if full != 0 || metadata == 0 {
			t.Errorf("%s read %d plan documents and %d metadata projections", name, full, metadata)
		}
	}
}

// Get still rejects what it cannot vouch for: a stored readiness that does
// not decode strictly fails as an internal error, and one that decodes but
// breaks the closed Readiness schema fails the Experiment response validation.
func TestSharedPostgresExperimentViewRejectsMalformedStoredReadiness(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "workflow")
	corrupt := func(readiness string) error {
		t.Helper()
		for _, statement := range []string{
			`ALTER TABLE eval_experiments DISABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments DISABLE TRIGGER eval_experiments_protect`,
			`UPDATE eval_experiments SET plan_readiness=convert_to('` + readiness + `','UTF8') WHERE experiment_id='` + e.ID + `'`,
			`ALTER TABLE eval_experiments ENABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments ENABLE TRIGGER eval_experiments_protect`,
		} {
			if _, err := h.pool.Exec(t.Context(), statement); err != nil {
				t.Fatal(err)
			}
		}
		_, err := h.service.Get(t.Context(), e.OwnerID, e.ID)
		return err
	}
	var domain *evaldomain.Error
	for _, malformed := range []string{`{"pins":[],"arms":[],"probe":true}`, `{"pins":[]}`, `[]`} {
		if err := corrupt(malformed); err == nil || errors.As(err, &domain) {
			t.Fatalf("malformed readiness %s: %v", malformed, err)
		}
	}
	invalid := `{"pins":[{"dimension":"source","requiredEqual":true,"baselineOrigin":"observed","candidateOrigin":"observed","status":"maybe"}],"arms":[]}`
	if err := corrupt(invalid); !evaldomain.IsCode(err, "eval_invalid") {
		t.Fatalf("schema-invalid readiness: %v", err)
	}
}
