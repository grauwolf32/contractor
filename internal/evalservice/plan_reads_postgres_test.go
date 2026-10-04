package evalservice

import (
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

// preparedAtNativeCeiling freezes a workflow plan with MaxNativeMembers members
// whose cases share one input artifact.
func (h *serviceHarness) preparedAtNativeCeiling(t *testing.T, capacity int) evalstore.Experiment {
	t.Helper()
	draft, data, _ := planInputs(t)
	source, _ := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).User(h.scope.OwnerID)
	payload := []byte("fixture ZIP is not executed")
	write, err := source.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: "shared-task"}, artifacts.Payload{MediaType: "application/zip", Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := source.Metadata(t.Context(), write.Ref)
	if err != nil {
		t.Fatal(err)
	}
	template := data.Cases[0]
	template.Task.Parameters = map[string]string{}
	template.Requires = []string{}
	template.Outputs = map[string]evaldomain.Output{}
	template.Inputs = map[string]evaldomain.Artifact{"task": {Scope: "user", ScopeID: h.scope.OwnerID, Namespace: write.Ref.Namespace, Name: write.Ref.Name, Revision: *write.Ref.Revision, SHA256: evaldomain.Digest(payload), MediaType: "application/zip", SizeBytes: meta.Size}}
	draft.CaseIDs = make([]string, evaldomain.MaxNativeMembers/2)
	data.Cases = make([]evaldomain.Case, len(draft.CaseIDs))
	for i := range draft.CaseIDs {
		draft.CaseIDs[i] = fmt.Sprintf("case-%03d", i)
		data.Cases[i] = template
		data.Cases[i].ID = draft.CaseIDs[i]
	}
	for i := range draft.Variants {
		draft.Variants[i].Kind, draft.Variants[i].Selector = "workflow", "audit-check@1"
	}
	draft.Repetitions = 1
	draft.Budgets.MaxMembers, draft.Budgets.MaxInFlight = evaldomain.MaxNativeMembers, capacity
	dataset := frozen(t, "DatasetInput", data)
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.PutDataset(t.Context(), h.scope, "r1", dataset, identity(t, "dataset", 0, dataset))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Native ceiling", ControlMode: "server", Draft: &draft})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "prepare"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateReady || e.Expected != evaldomain.MaxNativeMembers {
		t.Fatalf("prepare: state=%s members=%d diagnostic=%s", e.State, e.Expected, e.Diagnostic)
	}
	return e
}

func projectedMembers(t *testing.T, h *serviceHarness, id string) int {
	t.Helper()
	var n int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_member_projections WHERE experiment_id=$1 AND revision=projected_revision`, id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// Admission and collection read the frozen plan once per admitted and per
// collected member. The stored row was validated before its insert and cannot
// change, so a tick must not walk the 442 KB schema again for each read.
func TestPostgresNativeTickTrustsStoredPlanAtMemberCeiling(t *testing.T) {
	h := newHarness(t)
	const capacity = membersPerTick
	e := h.preparedAtNativeCeiling(t, capacity)
	h.command(t, e, "start")
	collected := projectedMembers(t, h, e.ID)
	// The probe field breaks the closed plan schema while every field the tick
	// decodes stays intact, so any revalidation of the stored plan fails the
	// tick. Only DDL can bypass eval_plans_immutable.
	for _, statement := range []string{
		`ALTER TABLE eval_frozen_plans DISABLE TRIGGER eval_plans_immutable`,
		`UPDATE eval_frozen_plans SET document=convert_to('{"probe":true,'||substr(convert_from(document,'UTF8'),2),'UTF8') WHERE experiment_id='` + e.ID + `'`,
		`ALTER TABLE eval_frozen_plans ENABLE TRIGGER eval_plans_immutable`,
	} {
		if _, err := h.pool.Exec(t.Context(), statement); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := evaldomain.Freeze(portablePlanSchema, h.planBytes(t, e)); !evaldomain.IsCode(err, "eval_invalid") {
		t.Fatalf("probe kept the stored plan schema-valid: %v", err)
	}
	tick(t, h.coordinator(t, "ceiling"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateRunning || e.Outstanding != capacity || count(t, h.pool, "eval_submissions") != capacity {
		t.Fatalf("tick admitted %d members in state %s, want %d", e.Outstanding, e.State, capacity)
	}
	if got := projectedMembers(t, h, e.ID) - collected; got < evaldomain.CollectionBatchSize-capacity {
		t.Fatalf("tick collected %d dirty members", got)
	}
}

func (h *serviceHarness) planBytes(t *testing.T, e evalstore.Experiment) []byte {
	t.Helper()
	plan, err := evalstore.NewPostgresStore(h.pool).FrozenPlan(t.Context(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	return plan.Document.Bytes()
}
