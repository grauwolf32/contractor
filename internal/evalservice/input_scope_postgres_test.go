package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/jackc/pgx/v5/pgxpool"
)

func scopeDataset(t *testing.T, h *serviceHarness, sourceScope, kind string) (evaldomain.Draft, evaldomain.DatasetInput, contracts.ArtifactRef) {
	t.Helper()
	draft, data := h.dataset(t, "workflow")
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool))
	source, _ := service.User(h.scope.OwnerID)
	scopeID := h.scope.OwnerID
	if sourceScope == "project" {
		source, _ = service.Project(h.scope.ProjectID)
		scopeID = h.scope.ProjectID
	}
	pack, err := source.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: "rules"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: fixture-rule\n")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range data.Cases {
		if kind == "audit" {
			data.Cases[i].Inputs = map[string]evaldomain.Artifact{}
			for name, payload := range map[string]string{
				"target":     `{"schema":"contractor.audit.web-target.v1","url":"https://example.test/"}`,
				"engagement": `{"schema":"contractor.engagement.v1","egress":{"routes":[{"origin":"https://example.test","methods":["GET"],"pathPrefixes":["/"],"addressRanges":["192.168.56.0/24"]}],"exclusions":[]},"identities":[{"id":"anonymous","kind":"anonymous","origins":["https://example.test"]}]}`,
			} {
				media := "application/json"
				if name == "engagement" {
					media = "application/vnd.contractor.engagement+json"
				}
				write, err := source.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: name + data.Cases[i].ID}, artifacts.Payload{MediaType: media, Data: []byte(payload)}, nil)
				if err != nil {
					t.Fatal(err)
				}
				data.Cases[i].Inputs[name] = evaldomain.Artifact{Scope: sourceScope, ScopeID: scopeID, Namespace: write.Ref.Namespace, Name: write.Ref.Name, Revision: *write.Ref.Revision,
					SHA256: evaldomain.Digest([]byte(payload)), MediaType: media, SizeBytes: int64(len(payload))}
			}
		}
		if sourceScope != "absent" {
			data.Cases[i].Inputs["rules"] = evaldomain.Artifact{Scope: sourceScope, ScopeID: scopeID,
				Namespace: pack.Ref.Namespace, Name: pack.Ref.Name, Revision: *pack.Ref.Revision,
				SHA256: evaldomain.Digest([]byte("id: fixture-rule\n")), MediaType: pack.MediaType, SizeBytes: int64(len("id: fixture-rule\n"))}
		}
	}
	for i := range draft.Variants {
		draft.Variants[i].Selector = "nuclei-target@1"
		draft.Variants[i].Parameters = map[string]string{"target": "https://example.test/"}
		if kind == "audit" {
			draft.Variants[i].Kind = "audit"
			draft.Variants[i].Selector = "web-scan@1"
			draft.Variants[i].Parameters = nil
		}
		if sourceScope != "absent" {
			draft.Variants[i].InputMapping = map[string]string{"rules": "rulepack"}
		}
	}
	doc := frozen(t, "DatasetInput", data)
	if err := h.service.tx(t.Context(), func(s *evalstore.Store) error {
		_, err := s.PutDataset(t.Context(), h.scope, "r2", doc, identity(t, "scope-dataset", 0, doc))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	draft.Dataset.Revision = "r2"
	return draft, data, pack.Ref
}

func scopeFixture(t *testing.T, h *serviceHarness, sourceScope, kind string) (evalstore.Experiment, contracts.ArtifactRef) {
	t.Helper()
	draft, _, pack := scopeDataset(t, h, sourceScope, kind)
	create := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "Input source scopes", ControlMode: "server", Draft: &draft})
	receipt, err := h.service.Create(t.Context(), h.scope, create, identity(t, "scope-create", 0, create))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	e := h.get(t, ref.ExperimentID)
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "scope-prepare"))
	return h.get(t, e.ID), pack
}

func TestSharedPostgresEvalUserRulepackCreatesRun(t *testing.T) {
	for _, mode := range []string{"user", "absent"} {
		t.Run(mode, func(t *testing.T) {
			h := newHarness(t)
			e, pack := scopeFixture(t, h, mode, "workflow")
			if e.State != evaldomain.StateReady {
				t.Fatalf("prepared state=%s diagnostic=%s", e.State, e.Diagnostic)
			}
			if mode == "user" {
				user, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).User(h.scope.OwnerID)
				if _, err := user.Write(t.Context(), contracts.ArtifactRef{Namespace: pack.Namespace, Name: pack.Name}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: newer-rule\n")}, pack.Revision); err != nil {
					t.Fatal(err)
				}
			}
			h.driver.Runs = h.run.RunCreator
			h.command(t, e, "start")
			c := h.coordinator(t, "scope-run")
			tick(t, c)
			tick(t, c)
			assertScopeRun(t, h, pack, mode != "absent")
		})
	}
}

func assertScopeRun(t *testing.T, h *serviceHarness, original contracts.ArtifactRef, withPack bool) string {
	t.Helper()
	if got := count(t, h.pool, "workflow_runs"); got != 1 {
		t.Fatalf("created %d Runs, want one", got)
	}
	var id string
	if err := h.pool.QueryRow(t.Context(), `SELECT run_id FROM workflow_runs`).Scan(&id); err != nil {
		t.Fatal(err)
	}
	store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).Run(id)
	for slot, want := range map[string]string{"task": "fixture ZIP is not executed", "rulepack": "id: fixture-rule\n"} {
		read, err := store.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: slot})
		if slot == "rulepack" && !withPack {
			if !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("absent pack: %v", err)
			}
			continue
		}
		if err != nil || string(read.Payload.Data) != want {
			t.Fatalf("%s fork: %s %v", slot, read.Payload.Data, err)
		}
		edges, err := store.ListLineage(t.Context(), read.Ref, artifacts.LineagePageQuery{Limit: 10})
		if err != nil || len(edges) != 1 {
			t.Fatalf("%s lineage: %+v %v", slot, edges, err)
		}
		edge := edges[0]
		if slot == "rulepack" {
			if edge.SourceScope != artifacts.ScopeUser || edge.SourceScopeID != h.scope.OwnerID || edge.Source.Namespace != original.Namespace || edge.Source.Name != original.Name || edge.Source.Revision == nil || *edge.Source.Revision != *original.Revision {
				t.Fatalf("rulepack source identity changed: %+v", edge)
			}
		} else if edge.SourceScope != artifacts.ScopeProject || edge.SourceScopeID != h.scope.ProjectID {
			t.Fatalf("ordinary input source changed: %+v", edge)
		}
	}
	var copied int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM artifact_bindings WHERE scope_kind='user' AND namespace <> 'fixtures'`).Scan(&copied); err != nil || copied != 0 {
		t.Fatalf("unowned UserScope staging: %d %v", copied, err)
	}
	return id
}

func TestSharedPostgresEvalProjectRulepackIsUnsupportedBeforeSubmission(t *testing.T) {
	h := newHarness(t)
	draft, data, _ := scopeDataset(t, h, "project", "workflow")
	for _, variant := range draft.Variants {
		pre, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, variant, data.Cases)
		if err != nil {
			t.Fatal(err)
		}
		for _, eligibility := range pre.Cases {
			if eligibility.State != "unsupported" || eligibility.Reason == nil || !strings.Contains(*eligibility.Reason, "UserScope") {
				t.Fatalf("scope mismatch eligibility: %+v", eligibility)
			}
		}
	}
	if count(t, h.pool, "workflow_runs") != 0 || count(t, h.pool, "eval_suboperations") != 0 {
		t.Fatal("preflight created work")
	}
}

func externalScopeFixture(t *testing.T, h *serviceHarness, kind, scope string) (evalstore.Experiment, contracts.ArtifactRef, string) {
	t.Helper()
	draft, data, pack := scopeDataset(t, h, scope, kind)
	pre := map[string]Preflight{}
	for _, v := range draft.Variants {
		p, err := h.resolver.Resolve(t.Context(), h.scope.OwnerID, v, data.Cases)
		if err != nil {
			t.Fatal(err)
		}
		pre[v.ID] = p
	}
	bundle, err := BuildPlan("external-scope", time.Date(2026, 10, 25, 0, 0, 0, 0, time.UTC), draft, data, pre)
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
	for _, m := range manifest.Members {
		recipes = append(recipes, evaldomain.MemberRecipe{MemberID: m.MemberID, Case: bundle.Cases[m.MemberID]})
	}
	doc := frozen(t, "CreateExperiment", evaldomain.CreateExperiment{Name: "External source scopes", ControlMode: "external", Registration: &evaldomain.ExternalRegistration{
		SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: bundle.Plan.Digest(), Manifest: manifest,
		Source: evaldomain.Source{System: "fixture", ID: "source-scopes"}, Variants: draft.Variants, Recipes: recipes,
		Checks: setup.Checks, Comparison: draft.Comparison, Budgets: draft.Budgets}})
	receipt, err := h.service.Create(t.Context(), h.scope, doc, identity(t, "external-scope-create", 0, doc))
	if err != nil {
		t.Fatal(err)
	}
	ref, err := receipt.Experiment()
	if err != nil {
		t.Fatal(err)
	}
	submission := evaldomain.Submission{PlanSHA256: bundle.Plan.Digest()}
	body := frozen(t, "Submission", submission)
	if _, err := h.service.Submit(t.Context(), h.scope, ref.ExperimentID, manifest.Members[0].MemberID, submission, identity(t, "external-scope-submit", 0, body)); err != nil {
		t.Fatal(err)
	}
	return h.get(t, ref.ExperimentID), pack, manifest.Members[0].MemberID
}

func TestSharedPostgresEvalSourceScopesForNativeAndExternalMembers(t *testing.T) {
	for _, test := range []struct{ kind, control string }{{"workflow", "external"}, {"audit", "external"}, {"audit", "server"}} {
		t.Run(test.kind+"-"+test.control, func(t *testing.T) {
			h := newHarness(t)
			h.driver.Runs, h.driver.Audits = h.run.RunCreator, h.audit.AuditExecutor
			sourceScope := "user"
			if test.kind == "audit" {
				sourceScope = "project"
			}
			c := h.coordinator(t, "external-scope")
			var e evalstore.Experiment
			var pack contracts.ArtifactRef
			var member string
			if test.control == "external" {
				e, pack, member = externalScopeFixture(t, h, test.kind, sourceScope)
			} else {
				e, pack = scopeFixture(t, h, sourceScope, test.kind)
				h.command(t, e, "start")
				tick(t, c)
				if err := h.pool.QueryRow(t.Context(), `SELECT member_id FROM eval_submissions WHERE experiment_id=$1`, e.ID).Scan(&member); err != nil {
					t.Fatal(err)
				}
			}
			tick(t, c)
			tick(t, c)
			if test.kind == "workflow" {
				assertScopeRun(t, h, pack, true)
				return
			}
			sub, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Submission(t.Context(), h.scope.OwnerID, e.ID, member)
			if err != nil || sub.ExecutionID == nil {
				t.Fatalf("Audit submission: %+v %v", sub, err)
			}
			audit, err := h.audit.Get(t.Context(), h.scope.OwnerID, *sub.ExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			baseline, err := auditservice.DecodeBaseline(audit.BaselineSnapshot)
			if err != nil {
				t.Fatal(err)
			}
			store, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).Project(audit.ProjectID)
			read, err := store.ReadExact(t.Context(), baseline.Inputs["rulepack"])
			if err != nil || string(read.Payload.Data) != "id: fixture-rule\n" || audit.ProjectID == h.scope.ProjectID {
				t.Fatalf("Audit must stage its exact rulepack in its execution Project: %v", err)
			}
		})
	}
}

type loseSourceBeforeRunCreate struct {
	RunCreator
	pool     *pgxpool.Pool
	observed error
}

func (r *loseSourceBeforeRunCreate) CreatePublic(ctx context.Context, p runservice.PublicCreateParams) (runservice.CreateResult, error) {
	ref := p.Inputs["rulepack"]
	_, err := r.pool.Exec(ctx, `DELETE FROM artifact_bindings WHERE scope_kind='user' AND scope_id=$1 AND namespace=$2 AND name=$3`, p.OwnerID, ref.Namespace, ref.Name)
	if err != nil {
		return runservice.CreateResult{}, err
	}
	result, err := r.RunCreator.CreatePublic(ctx, p)
	r.observed = err
	return result, err
}

func TestSharedPostgresEvalMissingRulepackAfterIntentCancelsAndPurges(t *testing.T) {
	h := newHarness(t)
	e, _ := scopeFixture(t, h, "user", "workflow")
	creator := &loseSourceBeforeRunCreate{RunCreator: h.run.RunCreator, pool: h.pool}
	h.driver.Runs = creator
	h.command(t, e, "start")
	c := h.coordinator(t, "missing-source")
	tick(t, c)
	tick(t, c)
	if !errors.Is(creator.observed, artifacts.ErrArtifactNotFound) {
		t.Fatalf("actual Run admission error=%v", creator.observed)
	}
	var rejected int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_suboperations WHERE kind='run-create' AND state='rejected' AND convert_from(response,'UTF8') LIKE '%eval_evidence_unavailable%'`).Scan(&rejected); err != nil || rejected != 1 {
		t.Fatalf("durable exact-input rejection: %d %v", rejected, err)
	}
	var unresolved int
	if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_suboperations WHERE kind='run-create' AND state='intent'`).Scan(&unresolved); err != nil || unresolved != 0 {
		t.Fatalf("unresolved definite create rejection: %d %v", unresolved, err)
	}
	h.command(t, h.get(t, e.ID), "cancel")
	for n := 0; n < 4; n++ {
		tick(t, c)
	}
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateCancelled || e.Outstanding != 0 || count(t, h.pool, "workflow_runs") != 0 {
		t.Fatalf("rejected creation did not drain: state=%s outstanding=%d", e.State, e.Outstanding)
	}
	remove := frozen(t, "Delete", map[string]any{})
	if _, err := h.service.Delete(t.Context(), h.scope, e.ID, identity(t, "delete-source-eval", e.Revision, remove)); err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 4; n++ {
		tick(t, c)
	}
	if _, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Get(t.Context(), h.scope.OwnerID, e.ID); !notFound(err) {
		t.Fatalf("rejected member blocked Eval purge: %v", err)
	}
}

func TestSharedPostgresEvalRulepackExactReplayAndCleanup(t *testing.T) {
	h := newHarness(t)
	e, pack := scopeFixture(t, h, "user", "workflow")
	h.command(t, e, "start")
	c := h.coordinator(t, "before-restart")
	tick(t, c)
	tick(t, c) // Ordinary Run commits; the harness loses its response.
	id := assertScopeRun(t, h, pack, true)
	user, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(h.pool)).User(h.scope.OwnerID)
	advanced, err := user.Write(t.Context(), contracts.ArtifactRef{Namespace: pack.Namespace, Name: pack.Name}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: replacement-rule\n")}, pack.Revision)
	if err != nil {
		t.Fatal(err)
	}
	h.command(t, h.get(t, e.ID), "cancel")
	c = h.coordinator(t, "after-restart")
	tick(t, c)
	h.finishRuns(t)
	tick(t, c)
	if got := h.get(t, e.ID); got.State != evaldomain.StateCancelled || got.Outstanding != 0 {
		t.Fatalf("replay cancellation: %s %d", got.State, got.Outstanding)
	}
	assertScopeRun(t, h, pack, true)
	if err := rundeletion.New(h.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(t.Context(), h.scope.OwnerID, id); err != nil {
		t.Fatal(err)
	}
	read, err := user.Read(t.Context(), advanced.Ref)
	if err != nil || string(read.Payload.Data) != "id: replacement-rule\n" {
		t.Fatalf("Run deletion touched owner source: %v", err)
	}
}
