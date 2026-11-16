package evalservice

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func externalPinFixture(t *testing.T, h *serviceHarness) evaldomain.ExternalRegistration {
	t.Helper()
	draft, data := h.dataset(t, "workflow")
	source := evaldomain.Digest([]byte("producer plan; not a locally observed digest"))
	reg := evaldomain.ExternalRegistration{
		SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: source,
		Manifest: evaldomain.PublicPlan{SchemaVersion: "playground.public-projection/v1", SourceSchemaVersion: "playground.plan/v1", SourceRecordSHA256: source, ExperimentID: "external-pins", CreatedAt: "2026-11-16T00:00:00Z"},
		Source:   evaldomain.Source{System: "fixture", ID: "external-pins"}, Variants: draft.Variants,
		Checks:     []evaldomain.Check{{ID: "domain", Evaluator: "fixture-domain@1", ImplementationSHA256: evaldomain.Digest([]byte("external scorer")), Required: true}},
		Comparison: evaldomain.Comparison{Baseline: "a", Candidate: "b", RequiredEqual: []string{"source", "tasks"}, AllowedDifferences: []string{"instructions"}, Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}, Budgets: draft.Budgets,
	}
	// Each arm sees the same cases, but first encounters them in opposite order.
	for i, v := range reg.Variants {
		cases := slices.Clone(data.Cases)
		if i == 1 {
			slices.Reverse(cases)
		}
		for _, c := range cases {
			c.Task.Objective = "Inspect exact material " + c.ID
			for sample := 1; sample <= 2; sample++ {
				id, err := evaldomain.MemberID(reg.Manifest.ExperimentID, "pins", c.ID, sample, v.ID)
				if err != nil {
					t.Fatal(err)
				}
				reg.Manifest.Members = append(reg.Manifest.Members, evaldomain.PublicMember{MemberID: id, SuiteID: "pins", CaseID: c.ID, Sample: sample, VariantID: v.ID, CaseSHA256: evaldomain.Digest([]byte("producer case " + c.ID)), BindingSHA256: evaldomain.Digest([]byte("producer binding " + v.ID)), Eligibility: "eligible"})
				reg.Recipes = append(reg.Recipes, evaldomain.MemberRecipe{MemberID: id, Case: c})
			}
		}
	}
	return reg
}

func TestSharedPostgresExternalObservedPinsUseOrdinaryComparison(t *testing.T) {
	h := newHarness(t)
	reg := externalPinFixture(t, h)
	e := registerPinFixture(t, h, reg, "external-pins-comparison")
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	c := h.coordinator(t, "external-pins-executions")
	submission := evaldomain.Submission{PlanSHA256: reg.SourcePlanSHA256}
	body := frozen(t, "Submission", submission)
	stages := map[string]string{}
	for i, declared := range reg.Manifest.Members {
		if _, err := h.service.Submit(t.Context(), h.scope, e.ID, declared.MemberID, submission, identity(t, fmt.Sprintf("pins-submit-%d", i), 0, body)); err != nil {
			t.Fatal(err)
		}
		for range 4 {
			tick(t, c)
			finishRunsWithStageMetrics(t, h, stages)
		}
		member, err := store.Member(t.Context(), e.OwnerID, e.ID, declared.MemberID)
		if err != nil {
			t.Fatal(err)
		}
		row, err := store.ExecutionObservation(t.Context(), e.OwnerID, e.ID, member.MemberID)
		if err != nil {
			t.Fatal(err)
		}
		execution := executionView(row)
		inventory, err := store.Inventory(t.Context(), e.OwnerID, e.ID, member.MemberID)
		if err != nil {
			t.Fatal(err)
		}
		usage, err := observedUsage(t.Context(), h.pool, e.OwnerID, member.MemberID, execution, inventory, artifactpgtest.NewStorage())
		if err != nil {
			t.Fatal(err)
		}
		outputs, err := collectOutputs(t.Context(), h.pool, e.OwnerID, member, execution, inventory, artifactpgtest.NewStorage())
		if err != nil {
			t.Fatal(err)
		}
		result := frozen(t, "ResultInput", evaldomain.ResultInput{
			SchemaVersion: evaldomain.ResultSchemaVersion, PlanSHA256: reg.SourcePlanSHA256, MemberID: member.MemberID, Source: reg.Source,
			Execution: execution, Usage: usage, Collection: outputs.Collection, Outputs: outputs.Outputs, Evidence: outputs.Evidence,
		})
		if _, err := h.service.PutRecord(t.Context(), h.scope, e.ID, member.MemberID, result, identity(t, fmt.Sprintf("pins-result-%d", i), 0, result)); err != nil {
			t.Fatal(err)
		}
		check := reg.Checks[0]
		assessment := frozen(t, "AssessmentInput", evaldomain.AssessmentInput{
			SchemaVersion: evaldomain.AssessmentSchemaVersion,
			Source:        evaldomain.AssessmentSource{Kind: "external", ProducerID: "fixture-domain", RecordSHA256: evaldomain.Digest([]byte("fixture decision"))},
			ResultSHA256:  result.Digest(), Checks: []evaldomain.CheckResult{{ID: check.ID, Evaluator: check.Evaluator, ImplementationSHA256: check.ImplementationSHA256, Status: "pass", Reason: "Fixture domain decision", EvidenceRefs: []string{}}},
		})
		if _, err := h.service.PutRecord(t.Context(), h.scope, e.ID, member.MemberID, assessment, identity(t, fmt.Sprintf("pins-assessment-%d", i), 0, assessment)); err != nil {
			t.Fatal(err)
		}
		record, err := store.LatestRecord(t.Context(), e.OwnerID, e.ID, member.MemberID, evaldomain.RecordKindAssessment)
		if err != nil || record == nil {
			t.Fatalf("assessment: %v, %v", record, err)
		}
		selection := evaldomain.SelectionInput{PlanSHA256: reg.SourcePlanSHA256, Selections: []evaldomain.SelectionEntry{{MemberID: member.MemberID, ResultSHA256: result.Digest(), AssessmentSHA256: &record.SHA256}}}
		selectionDoc := frozen(t, "SelectionInput", selection)
		e = h.get(t, e.ID)
		if _, err := h.service.Select(t.Context(), h.scope, e.ID, selection, identity(t, fmt.Sprintf("pins-select-%d", i), e.Revision, selectionDoc)); err != nil {
			t.Fatal(err)
		}
	}
	h.command(t, h.get(t, e.ID), "finalize")
	for range 4 {
		tick(t, c)
	}
	view, err := store.LatestView(t.Context(), e.OwnerID, e.ID)
	if err != nil || view == nil || !view.PinsVerified || view.Summary.CompleteTokenPairs != 4 || view.Summary.CompleteQualityPairs != 4 || view.Summary.Conclusion != "pass" {
		t.Fatalf("ordinary comparison: %+v, %v", view, err)
	}
	if len(stages) != len(reg.Manifest.Members) || count(t, h.pool, "workflow_runs") != len(reg.Manifest.Members) {
		t.Fatal("comparison did not use every ordinary Run exactly once")
	}
}

func TestSharedPostgresExternalPinsRejectUnverifiedInputsBeforePublication(t *testing.T) {
	for _, test := range []struct {
		name   string
		code   string
		change func(evaldomain.Artifact) evaldomain.Artifact
	}{
		{"missing", "eval_evidence_unavailable", func(a evaldomain.Artifact) evaldomain.Artifact { a.Name = "missing"; return a }},
		{"foreign", "eval_not_found", func(a evaldomain.Artifact) evaldomain.Artifact { a.ScopeID = "another-owner"; return a }},
		{"digest", "eval_pin_mismatch", func(a evaldomain.Artifact) evaldomain.Artifact {
			a.SHA256 = evaldomain.Digest([]byte("producer claim"))
			return a
		}},
		{"size", "eval_pin_mismatch", func(a evaldomain.Artifact) evaldomain.Artifact { a.SizeBytes++; return a }},
		{"media", "eval_pin_mismatch", func(a evaldomain.Artifact) evaldomain.Artifact { a.MediaType = "application/json"; return a }},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t)
			reg := externalPinFixture(t, h)
			for i := range reg.Recipes {
				for role, ref := range reg.Recipes[i].Case.Inputs {
					reg.Recipes[i].Case.Inputs[role] = test.change(ref)
				}
			}
			doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Unverified inputs", ControlMode: "external", Registration: &reg})
			if _, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "unverified-create", 0, doc)); !evaldomain.IsCode(err, test.code) {
				t.Fatalf("input rejection = %v, want %s", err, test.code)
			}
			if count(t, h.pool, "eval_experiments") != 0 || count(t, h.pool, "eval_members") != 0 || count(t, h.pool, "workflow_runs") != 0 {
				t.Fatal("unverified input partially published or dispatched")
			}
		})
	}
}

func TestSharedPostgresExternalFrozenBindingsAreNeverUpgraded(t *testing.T) {
	h := newHarness(t)
	reg := externalPinFixture(t, h)
	// Freeze a prior config-only binding through the ordinary store. This is a
	// retained resource fixture, not a production migration or fallback reader.
	resources := []evalstore.PlanResource{}
	original := map[string][]byte{}
	for _, v := range reg.Variants {
		seen := map[string]bool{}
		cases := []evaldomain.Case{}
		for _, r := range reg.Recipes {
			if !seen[r.Case.ID] {
				seen[r.Case.ID] = true
				cases = append(cases, r.Case)
			}
		}
		preflight, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, v, cases)
		if err != nil {
			t.Fatal(err)
		}
		doc, err := bindingDocument(v, preflight)
		if err != nil {
			t.Fatal(err)
		}
		path := "bindings/" + v.ID + ".json"
		resources = append(resources, evalstore.PlanResource{Path: path, Document: doc})
		original[path] = doc.Bytes()
	}
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Retained binding", ControlMode: "external", Registration: &reg})
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	if err := h.service.tx(t.Context(), func(st *evalstore.Store) error {
		_, err := st.Create(t.Context(), evalstore.CreateParams{Scope: h.scope, ID: "retained-external", PortableID: reg.Manifest.ExperimentID, Document: doc, Mutation: identity(t, "retained-create", 0, doc), Resources: resources})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	e := h.get(t, "retained-external")
	tick(t, h.coordinator(t, "retained-bindings"))
	view, err := store.LatestView(t.Context(), e.OwnerID, e.ID)
	if err != nil || view == nil || view.PinsVerified {
		t.Fatalf("retained provenance upgraded: %+v, %v", view, err)
	}
	for path, before := range original {
		after, err := store.PlanResource(t.Context(), e.OwnerID, e.ID, path)
		if err != nil || !bytes.Equal(before, after.Document.Bytes()) {
			t.Fatalf("retained resource changed: %s, %v", path, err)
		}
	}
}

func registerPinFixture(t *testing.T, h *serviceHarness, reg evaldomain.ExternalRegistration, key string) evalstore.Experiment {
	t.Helper()
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "External observed pins", ControlMode: "external", Registration: &reg})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, key, 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	return h.get(t, ref.ExperimentID)
}

func TestSharedPostgresExternalLocalSourceAndTaskPinsAreObserved(t *testing.T) {
	h := newHarness(t)
	reg := externalPinFixture(t, h)
	e := registerPinFixture(t, h, reg, "external-pins-create")
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	plan, err := store.FrozenPlan(t.Context(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	public, err := h.service.Get(t.Context(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if public.Readiness != nil {
		t.Fatal("external registration acquired native readiness")
	}
	verified, err := verifiedComparisonPins(t.Context(), store, e, plan, reg.Comparison)
	if err != nil || !verified {
		t.Fatalf("locally verified source/tasks are not comparable: %v, %v", verified, err)
	}
	for _, v := range reg.Variants {
		resource, err := store.PlanResource(t.Context(), e.OwnerID, e.ID, "bindings/"+v.ID+".json")
		if err != nil {
			t.Fatal(err)
		}
		var binding portableBinding
		if err := json.Unmarshal(resource.Document.Bytes(), &binding); err != nil {
			t.Fatal(err)
		}
		for _, dim := range []string{"source", "tasks"} {
			p := binding.Settings.ObservedPins[dim]
			if p.Origin != "observed" || p.Value == nil {
				t.Fatalf("%s: %+v", dim, p)
			}
		}
		for _, dim := range []string{"model-revision", "runtime-build", "tool-docstrings"} {
			if binding.Settings.ObservedPins[dim].Origin != "unavailable" {
				t.Fatalf("promoted unavailable %s", dim)
			}
		}
		for _, declared := range reg.Manifest.Members {
			m, err := store.Member(t.Context(), e.OwnerID, e.ID, declared.MemberID)
			if err != nil {
				t.Fatal(err)
			}
			if m.VariantID == v.ID && m.BindingSHA256 != evaldomain.Digest([]byte("producer binding "+v.ID)) {
				t.Fatal("rewrote attributed binding")
			}
		}
	}
	c := h.coordinator(t, "external-pins-view")
	tick(t, c)
	view, err := store.LatestView(t.Context(), e.OwnerID, e.ID)
	if err != nil || view == nil || !view.PinsVerified {
		t.Fatalf("native comparison view pins: %+v, %v", view, err)
	}
	// No execution is needed to verify provenance, and preparation stays read-only.
	if count(t, h.pool, "workflow_runs") != 0 {
		t.Fatal("registration dispatched model work")
	}
}
