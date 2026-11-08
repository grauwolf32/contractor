package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

const validEvalAssertions = `{"schema":"contractor.audit.assertions.v1","items":[{"key":"source-check","version":"1","statement":"Inspect retained source","scope":{},"grounds":"Authored requirement","applicability":"all","allowed_methods":["source-review"],"required_evidence":[]}]}`

// observeAuditStarts delegates to the actual PostgreSQL-backed Audit service.
// Only the uncertainty tests inject a failure before, or after, its acceptance.
type observeAuditStarts struct {
	AuditExecutor
	cause    error
	lose     bool
	params   []auditservice.StartParams
	results  []auditservice.StartedAudit
	failures []error
}

func (a *observeAuditStarts) Start(ctx context.Context, p auditservice.StartParams) (auditservice.StartedAudit, error) {
	a.params = append(a.params, p)
	if a.cause != nil {
		return auditservice.StartedAudit{}, a.cause
	}
	result, err := a.AuditExecutor.Start(ctx, p)
	a.results = append(a.results, result)
	a.failures = append(a.failures, err)
	if err == nil && a.lose {
		a.lose = false
		return auditservice.StartedAudit{}, errResponseLost
	}
	return result, err
}

// assertionExperiment retains exact current-schema bytes in the ordinary
// ArtifactStore, freezes an actual catalog checking program and uses capacity
// one. The external caller explicitly submits members as capacity is released.
func assertionExperiment(t *testing.T, h *serviceHarness, payload, control string) (evalstore.Experiment, []string, string) {
	t.Helper()
	draft, data := h.dataset(t, "audit")
	source, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).User(h.scope.OwnerID)
	write, err := source.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: "assertions"}, artifacts.Payload{MediaType: "application/json", Data: []byte(payload)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range data.Cases {
		data.Cases[i].Inputs = map[string]evaldomain.Artifact{"checklist": {
			Scope: "user", ScopeID: h.scope.OwnerID, Namespace: write.Ref.Namespace, Name: write.Ref.Name,
			Revision: *write.Ref.Revision, SHA256: evaldomain.Digest([]byte(payload)), MediaType: "application/json", SizeBytes: int64(len(payload)),
		}}
	}
	for i := range draft.Variants {
		draft.Variants[i].Selector = "test-assertions@1"
	}
	doc := frozen(t, "DatasetInput", data)
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.PutDataset(t.Context(), h.scope, "r2", doc, identity(t, "assertion-dataset", 0, doc))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	draft.Dataset.Revision = "r2"
	create := evaldomain.CreateExperiment{Name: "Strict assertions", ControlMode: evaldomain.ControlMode(control), Draft: &draft}
	var members []string
	var digest string
	if control == "external" {
		pre := map[string]Preflight{}
		for _, variant := range draft.Variants {
			pre[variant.ID], err = h.resolver.Resolve(t.Context(), h.scope.OwnerID, variant, data.Cases)
			if err != nil {
				t.Fatal(err)
			}
		}
		bundle, err := BuildPlan("assertion-source", time.Date(2026, 11, 7, 0, 0, 0, 0, time.UTC), draft, data, pre)
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
		if err := json.Unmarshal(bundle.Setup, &setup); err != nil {
			t.Fatal(err)
		}
		var recipes []evaldomain.MemberRecipe
		for _, member := range manifest.Members {
			members = append(members, member.MemberID)
			recipes = append(recipes, evaldomain.MemberRecipe{MemberID: member.MemberID, Case: bundle.Cases[member.MemberID]})
		}
		digest = bundle.Plan.Digest()
		create.Draft = nil
		create.Registration = &evaldomain.ExternalRegistration{
			SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: digest, Manifest: manifest,
			Source: evaldomain.Source{System: "fixture", ID: "assertion-source"}, Variants: draft.Variants, Recipes: recipes,
			Checks: setup.Checks, Comparison: draft.Comparison, Budgets: draft.Budgets,
		}
	}
	doc = frozen(t, "CreateExperiment", create)
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "assertion-create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	if control == "server" {
		h.command(t, e, "prepare")
		tick(t, h.coordinator(t, "assertion-prepare"))
		e = h.get(t, e.ID)
	}
	if e.State != evaldomain.StateReady || e.Expected != 8 || e.MaxInFlight != 1 {
		t.Fatalf("prepare = %s expected=%d capacity=%d diagnostic=%s", e.State, e.Expected, e.MaxInFlight, e.Diagnostic)
	}
	return e, members, digest
}

func TestSharedPostgresInvalidAuditAssertionsRejectAndDrain(t *testing.T) {
	for _, control := range []string{"server", "external"} {
		for _, source := range []struct {
			name, payload string
			domain        bool
		}{
			{name: "missing-grounds", payload: strings.Replace(validEvalAssertions, `"grounds":"Authored requirement",`, "", 1)},
			{name: "trailing-json", payload: validEvalAssertions + `{}`, domain: true},
		} {
			t.Run(control+"/"+source.name, func(t *testing.T) {
				h := newHarness(t)
				audits := &observeAuditStarts{AuditExecutor: h.audit.AuditExecutor}
				h.driver.Audits = audits
				e, members, digest := assertionExperiment(t, h, source.payload, control)
				c := h.coordinator(t, "assertion-controller")
				if control == "server" {
					h.command(t, e, "start")
				} else {
					// With capacity one each subsequent accepted POST demonstrates
					// that ordinary draft deletion released its predecessor's slot.
					for i, member := range members {
						submission := evaldomain.Submission{PlanSHA256: digest}
						body := frozen(t, "Submission", submission)
						if _, err := h.service.Submit(t.Context(), h.scope, e.ID, member, submission, identity(t, fmt.Sprintf("submit-%d", i), 0, body)); err != nil {
							t.Fatal(err)
						}
						for n := 0; n < 10; n++ {
							tick(t, c)
							h.purgeDeletingAuditDrafts(t)
							e = h.get(t, e.ID)
							if e.Outstanding == 0 {
								break
							}
						}
						if e.Outstanding != 0 {
							t.Fatalf("external member %d retained capacity", i)
						}
					}
					h.command(t, e, "finalize")
				}
				for n := 0; n < 80; n++ {
					tick(t, c)
					h.purgeDeletingAuditDrafts(t)
					e = h.get(t, e.ID)
					if e.State == evaldomain.StateFinished {
						break
					}
				}
				if e.State != evaldomain.StateFinished || e.Outstanding != 0 {
					t.Fatalf("invalid source did not drain: %s outstanding=%d", e.State, e.Outstanding)
				}
				if len(audits.params) != 8 {
					t.Fatalf("Audit Start calls = %d, want exactly one per member", len(audits.params))
				}
				for _, err := range audits.failures {
					classified, ok := apierror.Of(err)
					var validation *auditdomain.ValidationError
					if !ok || classified.ErrorKind() != apierror.AuditDocumentInvalid || classified.ErrorKind().Status() != 422 || classified.ErrorKind().Retryable() || errors.As(err, &validation) != source.domain {
						t.Fatalf("actual Audit source error = %T %v, want domain=%v", err, err, source.domain)
					}
				}
				var rejected, settled, intents int
				if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_suboperations WHERE kind='audit-start' AND state='rejected' AND convert_from(response,'UTF8') LIKE '%eval_not_ready%'`).Scan(&rejected); err != nil {
					t.Fatal(err)
				}
				if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions WHERE state='rejected' AND execution_id IS NULL`).Scan(&settled); err != nil {
					t.Fatal(err)
				}
				if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_suboperations WHERE state='intent'`).Scan(&intents); err != nil {
					t.Fatal(err)
				}
				if rejected != 8 || settled != 8 || intents != 0 || count(t, h.pool, "audits") != 0 || count(t, h.pool, "workflow_runs") != 0 {
					t.Fatalf("rejected starts=%d settled=%d intents=%d; drafts and Runs must be empty", rejected, settled, intents)
				}
			})
		}
	}
}

func TestSharedPostgresUncertainAuditStartKeepsKeyForReplay(t *testing.T) {
	for _, test := range []struct {
		name     string
		cause    error
		accepted bool
	}{
		{name: "timeout", cause: context.DeadlineExceeded},
		{name: "unknown-io", cause: errors.New("fixture: unknown storage outcome")},
		{name: "unrelated-422", cause: apierror.New(apierror.AuditProfileUnsupported, "unrelated 422")},
		{name: "accepted-response-lost", cause: errResponseLost, accepted: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t)
			audits := &observeAuditStarts{AuditExecutor: h.audit.AuditExecutor, cause: test.cause}
			if test.accepted {
				audits.cause, audits.lose = nil, true
			}
			h.driver.Audits = audits
			e, _, _ := assertionExperiment(t, h, validEvalAssertions, "server")
			h.command(t, e, "start")
			c := h.coordinator(t, "uncertain-start")
			tick(t, c) // Admit only; execution starts on the next pass.
			if _, err := c.RunOnce(t.Context()); !errors.Is(err, test.cause) {
				t.Fatalf("uncertain start = %v, want original cause", err)
			}
			var state, key string
			if err := h.pool.QueryRow(t.Context(), `SELECT state, operation_key FROM eval_suboperations WHERE kind='audit-start'`).Scan(&state, &key); err != nil || state != "intent" {
				t.Fatalf("uncertain start journal = %s %v", state, err)
			}
			audits.cause = nil
			tick(t, c)
			if len(audits.params) != 2 || audits.params[0] != audits.params[1] || audits.params[1].IdempotencyKey != key {
				t.Fatalf("retry changed the journaled request: %+v", audits.params)
			}
			if test.accepted && (len(audits.results) != 2 || !audits.results[1].Replayed || audits.results[0].Audit.AuditID != audits.results[1].Audit.AuditID) {
				t.Fatalf("accepted start did not replay its receipt: %+v", audits.results)
			}
			if err := h.pool.QueryRow(t.Context(), `SELECT state FROM eval_suboperations WHERE kind='audit-start'`).Scan(&state); err != nil || state != "succeeded" || count(t, h.pool, "audits") != 1 {
				t.Fatalf("replayed start = %s, %v", state, err)
			}
		})
	}
}
