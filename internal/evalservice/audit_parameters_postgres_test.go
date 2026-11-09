package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type observeAuditCreates struct {
	AuditExecutor
	params  []auditservice.CreateDraftParams
	ids     []string
	created []bool
}

func (a *observeAuditCreates) CreateDraft(ctx context.Context, p auditservice.CreateDraftParams) (auditstore.Audit, bool, error) {
	a.params = append(a.params, p)
	v, created, err := a.AuditExecutor.CreateDraft(ctx, p)
	if err != nil {
		return v, created, err
	}
	a.ids = append(a.ids, v.AuditID)
	a.created = append(a.created, created)
	if len(a.params) == 1 {
		return auditstore.Audit{}, false, errResponseLost
	}
	return v, created, nil
}

func TestSharedPostgresAuditParametersSurviveLostCreateResponse(t *testing.T) {
	h := newHarness(t)
	audits := &observeAuditCreates{AuditExecutor: h.audit.AuditExecutor}
	h.driver.Audits = audits
	draft, data := h.dataset(t, "audit")
	want := map[string]string{"server": "https://variant.test", "operations": "getPet", "test_parameters": "id", "credential_id": "fixture-credential", "tags": "web"}
	for i := range data.Cases {
		data.Cases[i].Task.Objective = "frozen task objective"
		data.Cases[i].Task.Parameters = maps.Clone(want)
		data.Cases[i].Task.Parameters["server"] = "https://case.test"
		data.Cases[i].Task.Parameters["objective"] = "case objective"
		data.Cases[i].Task.Parameters["target"] = "scope target"
	}
	for i := range draft.Variants {
		draft.Variants[i].Selector = "test-parameters@1"
		draft.Variants[i].Parameters = map[string]string{"server": want["server"], "objective": "$task.objective"}
		pre, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, draft.Variants[i], data.Cases)
		if err != nil {
			t.Fatal(err)
		}
		for id, eligible := range pre.Cases {
			if eligible.State != "eligible" {
				t.Fatalf("case %s: %+v", id, eligible)
			}
		}
	}
	doc := frozen(t, "DatasetInput", data)
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.PutDataset(t.Context(), h.scope, "r2", doc, identity(t, "parameters-dataset", 0, doc))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	draft.Dataset.Revision = "r2"
	doc = frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Audit parameters", ControlMode: "server", Draft: &draft})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "parameters-create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "parameters-prepare"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateReady {
		t.Fatalf("prepare = %s: %s", e.State, e.Diagnostic)
	}
	h.command(t, e, "start")
	c := h.coordinator(t, "parameters-before-restart")
	tick(t, c)
	if _, err := c.RunOnce(t.Context()); !errors.Is(err, errResponseLost) {
		t.Fatalf("create = %v", err)
	}
	var state, key string
	var request []byte
	if err := h.pool.QueryRow(t.Context(), `SELECT state, operation_key, request FROM eval_suboperations WHERE kind='audit-create'`).Scan(&state, &key, &request); err != nil || state != "intent" {
		t.Fatalf("journal = %s: %v", state, err)
	}
	var retained struct{ Parameters map[string]string }
	if err := json.Unmarshal(request, &retained); err != nil || !maps.Equal(retained.Parameters, want) {
		t.Fatalf("retained launch parameters = %v, %v", retained.Parameters, err)
	}
	// Recreate the controller after the accepted response was lost. The durable
	// intent supplies both the original values and the ordinary creation key.
	tick(t, h.coordinator(t, "parameters-after-restart"))
	if len(audits.params) != 2 || audits.params[0].IdempotencyKey != key || audits.params[1].IdempotencyKey != key || audits.params[0].RequestDigest != audits.params[1].RequestDigest {
		t.Fatalf("creation replay changed identity: %+v", audits.params)
	}
	for _, p := range audits.params {
		if !maps.Equal(p.Parameters, want) || p.Scope.Target != "scope target" || p.Scope.AuthorizationScope != "" || p.Scope.Objective != "frozen task objective" {
			t.Fatalf("creation parameters/scope = %+v", p)
		}
	}
	if len(audits.created) != 2 || !audits.created[0] || audits.created[1] || audits.ids[0] != audits.ids[1] || count(t, h.pool, "audits") != 1 {
		t.Fatalf("create did not replay: %+v %+v", audits.created, audits.ids)
	}
	audit, err := audits.Get(t.Context(), h.scope.OwnerID, audits.ids[0])
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := auditservice.DecodeBaseline(audit.BaselineSnapshot)
	if err != nil || !maps.Equal(baseline.Parameters, want) {
		t.Fatalf("baseline = %+v, %v", baseline.Parameters, err)
	}
	program, err := auditdomain.ExecutionProgram(audits.params[0].PinnedProfile.Program, audit.BaselineSnapshot)
	if err != nil || !maps.Equal(program.Workflows["check"].Parameters, want) {
		t.Fatalf("Workflow binding = %+v, %v", program.Workflows["check"].Parameters, err)
	}
	if err := h.pool.QueryRow(t.Context(), `SELECT state FROM eval_suboperations WHERE kind='audit-create'`).Scan(&state); err != nil || state != "succeeded" {
		t.Fatalf("replayed journal = %s: %v", state, err)
	}
}
