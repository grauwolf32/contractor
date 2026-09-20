package public

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type evalAPIHarness struct {
	handler     http.Handler
	pool        *pgxpool.Pool
	service     *evalservice.Service
	resolver    *evalservice.Resolver
	coordinator *evalcoordinator.Coordinator
	contract    routers.Router
	serial      int
}

func newEvalAPIHarness(t *testing.T) *evalAPIHarness {
	t.Helper()
	pool := isolatedPublicPool(t, t.Context())
	catalog := loadPublicAuditConfiguration(t)
	credentialLookup := newFakeManagedCredentials()
	gateway, err := catalog.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"development-worker", "development-planner"} {
		credentialLookup.lookups[id] = config.CredentialMetadata{Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: gateway.Ref, Unrestricted: true}
	}
	credentials := runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) { return credentialLookup, nil })
	audits, err := auditservice.New(auditservice.Options{Pool: pool, Profiles: catalog, TransactionLLMCredentials: credentials, CredentialGuard: credentialLookup})
	if err != nil {
		t.Fatal(err)
	}
	resolver := &evalservice.Resolver{Pool: pool, Catalog: catalog, Credentials: credentials, Barrier: credentialLookup}
	h := &evalAPIHarness{pool: pool, resolver: resolver}
	fixture := newHandlerFixtureWithAuth(t, "../../../configs", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) {
		d.Config = catalog
		d.Audits = audits
		d.Projects = projectstore.NewPostgresStore(pool)
		d.Artifacts = artifacts.NewService(artifacts.NewPostgresRepository(pool))
		runs, err := runservice.New(runservice.Options{Runs: runstore.NewPostgresStore(pool), Workflows: catalog, LLMCredentials: credentialLookup, CredentialGuard: credentialLookup, RuntimeCredentials: d.RuntimeCredentials, Projects: d.Projects, SkillInitializationAvailable: true, PublicTransaction: func(ctx context.Context, fn func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return pg.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
				lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, credentials)
				if err != nil {
					return err
				}
				return fn(runstore.NewRunCreationPostgresStore(tx, lookup), artifacts.NewService(artifacts.NewPostgresRepository(tx)))
			})
		}})
		if err != nil {
			t.Fatal(err)
		}
		d.RunCreator = runs
		service, err := evalservice.New(evalservice.Options{Pool: pool, Resolver: resolver, Driver: &evalservice.Driver{Pool: pool, Runs: runs, Audits: audits}})
		if err != nil {
			t.Fatal(err)
		}
		h.service = service
		d.Evals = service
	})
	h.handler = fixture.handler
	h.coordinator, err = evalcoordinator.New(evalstore.NewPostgresStore(pool), h.service, evalcoordinator.Options{})
	if err != nil {
		t.Fatal(err)
	}
	h.contract, err = gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []struct {
		id, owner string
		kind      projectstore.Kind
	}{{"evaluation", "user-1", projectstore.KindEvaluation}, {"foreign", "user-2", projectstore.KindEvaluation}, {"ordinary", "user-1", projectstore.KindProject}} {
		_, _, err := projectstore.NewPostgresStore(pool).Create(t.Context(), projectstore.CreateParams{ProjectID: p.id, OwnerID: p.owner, Kind: p.kind, Name: p.id, IdempotencyKey: p.id, RequestDigest: evaldomain.Digest([]byte(p.id))})
		if err != nil {
			t.Fatal(err)
		}
	}
	return h
}
func (h *evalAPIHarness) request(t *testing.T, method, path string, body any, key, etag string, want int) *httptest.ResponseRecorder {
	t.Helper()
	var raw []byte
	var err error
	if body != nil {
		raw, err = json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
	}
	r := newPublicContractRequest(method, path, raw)
	if body != nil {
		r.Header.Set("Content-Type", "application/json")
	}
	if key != "" {
		r.Header.Set("Idempotency-Key", key)
	}
	if etag != "" {
		r.Header.Set("If-Match", etag)
	}
	response := serveAndValidatePublicContract(t, h.contract, h.handler, r, want < 400)
	if response.Code != want {
		t.Fatalf("%s %s: got %d want %d: %s", method, path, response.Code, want, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "PRIVATE_") {
		t.Fatal("generic Eval response disclosed private expected data")
	}
	return response
}
func apiDecode[T any](t *testing.T, r *httptest.ResponseRecorder) T {
	t.Helper()
	var out T
	if err := json.Unmarshal(r.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out
}
func evalFixture(t *testing.T, name string, out any) {
	t.Helper()
	b, err := os.ReadFile("../../../api/testdata/evals/valid/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, out); err != nil {
		t.Fatal(err)
	}
}
func (h *evalAPIHarness) dataset(t *testing.T, kind string) (evaldomain.Draft, evaldomain.DatasetInput) {
	t.Helper()
	var draft evaldomain.Draft
	var data evaldomain.DatasetInput
	evalFixture(t, "draft", &draft)
	evalFixture(t, "dataset", &data)
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	for i := range data.Cases {
		c := &data.Cases[i]
		c.Inputs = map[string]evaldomain.Artifact{}
		c.Requires = []string{}
		c.Outputs = map[string]evaldomain.Output{}
		c.Task.Parameters = map[string]string{}
		payloads := map[string]artifacts.Payload{"source": {MediaType: "text/plain", Data: []byte("fixture")}}
		if kind == "audit" {
			payloads["source"] = artifacts.Payload{MediaType: "application/zip", Data: []byte("fixture archive")}
			payloads["checklist"] = artifacts.Payload{MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check","version":"1","statement":"Review fixture.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`)}
		}
		for slot, payload := range payloads {
			written, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: c.ID + "-" + slot}, payload, nil)
			if err != nil {
				t.Fatal(err)
			}
			c.Inputs[slot] = evaldomain.Artifact{Scope: "user", ScopeID: "user-1", Namespace: written.Ref.Namespace, Name: written.Ref.Name, Revision: *written.Ref.Revision, SHA256: evaldomain.Digest(payload.Data), MediaType: payload.MediaType, SizeBytes: int64(len(payload.Data))}
		}
	}
	for i := range draft.Variants {
		draft.Variants[i].Kind = kind
		draft.Variants[i].Selector = "artifact-copy@1"
		if kind == "audit" {
			draft.Variants[i].Selector = "public-checklist@1"
		}
	}
	response := h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", data, "import", "", 201)
	metadata := apiDecode[evaldomain.Dataset](t, response)
	draft.Dataset.Revision = metadata.Revision
	replay := h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", data, "import", "", 201)
	if !bytes.Equal(response.Body.Bytes(), replay.Body.Bytes()) || replay.Header().Get("Idempotency-Replayed") != "true" {
		t.Fatal("dataset receipt replay changed")
	}
	return draft, data
}
func (h *evalAPIHarness) tick(t *testing.T) {
	t.Helper()
	if _, err := h.coordinator.RunOnce(t.Context()); err != nil {
		t.Fatal(err)
	}
}
func (h *evalAPIHarness) get(t *testing.T, id string) evalservice.ExperimentView {
	return apiDecode[evalservice.ExperimentView](t, h.request(t, "GET", "/v1/eval-experiments/"+id, nil, "", "", 200))
}
func (h *evalAPIHarness) command(t *testing.T, id, kind string) *httptest.ResponseRecorder {
	t.Helper()
	e := h.get(t, id)
	command := evaldomain.Command{Kind: evaldomain.CommandKind(kind)}
	if e.PlanSHA256 != nil {
		command.PlanSHA256 = *e.PlanSHA256
	}
	h.serial++
	status := 202
	if kind == "duplicate" {
		status = 201
	}
	return h.request(t, "POST", "/v1/eval-experiments/"+id+"/commands", command, fmt.Sprintf("command-%d", h.serial), strconv.Quote(strconv.FormatInt(e.Revision, 10)), status)
}

func (h *evalAPIHarness) finish(t *testing.T, kind string) {
	t.Helper()
	if kind == "workflow" {
		rows, err := h.pool.Query(t.Context(), `SELECT run_id FROM workflow_runs WHERE state='running'`)
		if err != nil {
			t.Fatal(err)
		}
		ids := []string{}
		for rows.Next() {
			var id string
			if err = rows.Scan(&id); err != nil {
				t.Fatal(err)
			}
			ids = append(ids, id)
		}
		rows.Close()
		if rows.Err() != nil {
			t.Fatal(rows.Err())
		}
		for _, id := range ids {
			if _, err = runstore.NewPostgresStore(h.pool).TransitionRun(t.Context(), id, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "fixture"}); err != nil {
				t.Fatal(err)
			}
		}
	} else {
		store := auditstore.NewPostgresStore(h.pool)
		claims, err := store.Claim(t.Context(), auditstore.ClaimParams{HolderID: "api-fixture", Lease: time.Minute, Limit: 100})
		if err != nil {
			t.Fatal(err)
		}
		for _, claim := range claims {
			a, err := store.Get(t.Context(), "user-1", claim.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if a.State == auditstore.AuditActive {
				a, err = store.TransitionClaimed(t.Context(), auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: a.Revision, ExpectedState: a.State, TargetState: auditstore.AuditFinalizing})
				if err != nil {
					t.Fatal(err)
				}
				_, err = store.TransitionClaimed(t.Context(), auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: a.Revision, ExpectedState: a.State, TargetState: auditstore.AuditFailed})
				if err != nil {
					t.Fatal(err)
				}
			}
			if err = store.ReleaseClaim(t.Context(), claim); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestEvalPostgresExternalWorkflowAndAuditSubmission(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		t.Run(kind, func(t *testing.T) {
			h := newEvalAPIHarness(t)
			draft, data := h.dataset(t, kind)
			pre := map[string]evalservice.Preflight{}
			for _, v := range draft.Variants {
				p, err := h.resolver.Resolve(t.Context(), "user-1", v, data.Cases)
				if err != nil {
					t.Fatal(err)
				}
				pre[v.ID] = p
			}
			bundle, err := evalservice.BuildPlan("external-api", time.Now(), draft, data, pre)
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
			recipes := []evaldomain.MemberRecipe{}
			for _, m := range manifest.Members {
				recipes = append(recipes, evaldomain.MemberRecipe{MemberID: m.MemberID, Case: bundle.Cases[m.MemberID]})
			}
			create := evaldomain.CreateExperiment{Name: "Independent producer", ControlMode: "external", Registration: &evaldomain.ExternalRegistration{SchemaVersion: "contractor.eval-registration/v1", SourcePlanSHA256: bundle.Plan.Digest(), Manifest: manifest, Source: evaldomain.Source{System: "fixture", ID: "independent-client"}, Variants: draft.Variants, Recipes: recipes, Checks: setup.Checks, Comparison: draft.Comparison, Budgets: draft.Budgets}}
			receipt := h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "create", "", 201)
			ref := apiDecode[struct {
				ID string `json:"experimentId"`
			}](t, receipt)
			h.tick(t)
			e := h.get(t, ref.ID)
			if e.State != "ready" || e.StartedAt != nil {
				t.Fatal("native coordinator dispatched external plan")
			}
			submissionPath := "/v1/eval-experiments/" + e.ID + "/members/" + manifest.Members[0].MemberID + "/submissions"
			body := evaldomain.Submission{PlanSHA256: *e.PlanSHA256}
			accepted := h.request(t, "POST", submissionPath, body, "submit", "", 202)
			h.tick(t)
			h.finish(t, kind)
			h.tick(t)
			if h.get(t, e.ID).State != "running" {
				t.Fatal("external producer was implicitly finalized")
			}
			h.command(t, e.ID, "finalize")
			h.tick(t)
			if h.get(t, e.ID).State != "finished" {
				t.Fatal("external finalization failed")
			}
			replay := h.request(t, "POST", submissionPath, body, "submit", "", 202)
			if !bytes.Equal(accepted.Body.Bytes(), replay.Body.Bytes()) {
				t.Fatal("submission replay response changed")
			}
			h.request(t, "POST", "/v1/eval-experiments/"+e.ID+"/members/"+manifest.Members[1].MemberID+"/submissions", body, "late", "", 409)
			members := h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/members", nil, "", "", 200)
			page := apiDecode[struct {
				Items []evalservice.MemberView `json:"items"`
			}](t, members)
			if len(page.Items) != 8 {
				t.Fatal("missing external members disappeared")
			}
			if kind == "workflow" {
				finished := h.get(t, e.ID)
				h.request(t, "DELETE", "/v1/eval-experiments/"+e.ID, map[string]any{}, "delete", strconv.Quote(strconv.FormatInt(finished.Revision, 10)), 202)
				h.tick(t)
				replay = h.request(t, "POST", submissionPath, body, "submit", "", 202)
				if !bytes.Equal(accepted.Body.Bytes(), replay.Body.Bytes()) {
					t.Fatal("purge lost accepted submission receipt")
				}
			}
		})
	}
}

func TestEvalPostgresAuthorizationCSRFStrictBodiesAndCAS(t *testing.T) {
	h := newEvalAPIHarness(t)
	draft, data := h.dataset(t, "workflow")
	create := evaldomain.CreateExperiment{Name: "Authorized", ControlMode: "server", Draft: &draft}
	receipt := h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "create", "", 201)
	ref := apiDecode[struct {
		ID string `json:"experimentId"`
	}](t, receipt)
	paths := []struct{ method, path string }{
		{"GET", "/v1/eval-capabilities"}, {"GET", "/v1/projects/evaluation/eval-datasets"}, {"POST", "/v1/projects/evaluation/eval-datasets"},
		{"GET", "/v1/projects/evaluation/eval-datasets/trace-small/revisions/" + draft.Dataset.Revision + "/cases"},
		{"POST", "/v1/projects/evaluation/eval-experiments"}, {"GET", "/v1/eval-experiments"}, {"GET", "/v1/eval-experiments/" + ref.ID}, {"PATCH", "/v1/eval-experiments/" + ref.ID}, {"DELETE", "/v1/eval-experiments/" + ref.ID},
		{"POST", "/v1/eval-experiments/" + ref.ID + "/commands"}, {"GET", "/v1/eval-experiments/" + ref.ID + "/commands/missing"}, {"GET", "/v1/eval-experiments/" + ref.ID + "/members"}, {"POST", "/v1/eval-experiments/" + ref.ID + "/members/" + strings.Repeat("a", 64) + "/submissions"},
	}
	for _, route := range paths {
		for _, token := range []string{"", "Bearer private-worker-credential"} {
			r := newPublicContractRequest(route.method, route.path, []byte(`{}`))
			if token == "" {
				r.Header.Del("Authorization")
			} else {
				r.Header.Set("Authorization", token)
			}
			w := httptest.NewRecorder()
			h.handler.ServeHTTP(w, r)
			if w.Code != 401 {
				t.Fatalf("%s %s accepted foreign credential: %d", route.method, route.path, w.Code)
			}
		}
		r := newPublicContractRequest("HEAD", route.path, nil)
		w := httptest.NewRecorder()
		h.handler.ServeHTTP(w, r)
		if w.Code != 405 {
			t.Fatalf("HEAD %s = %d", route.path, w.Code)
		}
	}
	login := newPublicContractRequest("POST", "/v1/auth/login", []byte(`{"username":"admin","password":"correct horse battery staple"}`))
	login.Header.Del("Authorization")
	login.Header.Set("Content-Type", "application/json")
	login.Header.Set("Origin", testBrowserOrigin)
	w := httptest.NewRecorder()
	h.handler.ServeHTTP(w, login)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	cookies := w.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatal("session cookie missing")
	}
	for _, route := range paths {
		if route.method == "GET" {
			continue
		}
		r := newPublicContractRequest(route.method, route.path, []byte(`{}`))
		r.Header.Del("Authorization")
		r.AddCookie(cookies[0])
		r.Header.Set("Origin", testBrowserOrigin)
		r.Header.Set("Content-Type", "application/json")
		out := httptest.NewRecorder()
		h.handler.ServeHTTP(out, r)
		if out.Code != 403 {
			t.Fatalf("%s skipped CSRF: %d", route.path, out.Code)
		}
	}
	h.request(t, "POST", "/v1/projects/foreign/eval-experiments", create, "foreign", "", 404)
	h.request(t, "POST", "/v1/projects/ordinary/eval-experiments", create, "ordinary", "", 404)
	h.request(t, "GET", "/v1/projects/foreign/eval-datasets", nil, "", "", 404)
	data.DatasetID = "foreign-input"
	for key, input := range data.Cases[0].Inputs {
		input.ScopeID = "user-2"
		data.Cases[0].Inputs[key] = input
	}
	h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", data, "foreign-input", "", 404)
	for _, raw := range []string{`null`, `{"kind":"prepare","kind":"start"}`, `{"kind":"prepare","unexpected":true}`, `{"kind":"prepare","planSha256":null}`} {
		r := newPublicContractRequest("POST", "/v1/eval-experiments/"+ref.ID+"/commands", []byte(raw))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Idempotency-Key", "bad")
		r.Header.Set("If-Match", `"1"`)
		out := httptest.NewRecorder()
		h.handler.ServeHTTP(out, r)
		if out.Code != 422 {
			t.Fatalf("strict JSON %s: %d %s", raw, out.Code, out.Body.String())
		}
	}
	path := "/v1/eval-experiments/" + ref.ID + "/commands"
	body := evaldomain.Command{Kind: "prepare"}
	h.request(t, "POST", path, body, "missing-cas", "", 428)
	h.request(t, "POST", path, body, "stale", `"2"`, 412)
	first := h.request(t, "POST", path, body, "prepare", `"1"`, 202)
	h.tick(t)
	replay := h.request(t, "POST", path, body, "prepare", `"1"`, 202)
	if !bytes.Equal(first.Body.Bytes(), replay.Body.Bytes()) {
		t.Fatal("command receipt replay was re-rendered from current state")
	}
	h.request(t, "POST", path, body, "prepare", `"2"`, 409)
}

func TestEvalPostgresBoundedCollectionsRejectStaleAndForeignCursors(t *testing.T) {
	h := newEvalAPIHarness(t)
	capsPath := "/v1/eval-capabilities?kind=workflow&limit=1"
	caps := apiDecode[struct {
		Bindings []struct {
			Selector string `json:"selector"`
			Kind     string `json:"kind"`
		} `json:"bindings"`
		Page evalPageInfo `json:"page"`
	}](t, h.request(t, "GET", capsPath, nil, "", "", 200))
	if len(caps.Bindings) != 1 || caps.Bindings[0].Kind != "workflow" || !strings.Contains(caps.Bindings[0].Selector, "@") || caps.Page.NextCursor == nil {
		t.Fatal("exact bounded Workflow capabilities missing")
	}
	h.request(t, "GET", capsPath+"&cursor="+url.QueryEscape(*caps.Page.NextCursor), nil, "", "", 200)
	h.request(t, "GET", "/v1/eval-capabilities?kind=audit&limit=1&cursor="+url.QueryEscape(*caps.Page.NextCursor), nil, "", "", 422)
	draft, data := h.dataset(t, "workflow")
	create := evaldomain.CreateExperiment{Name: "One", ControlMode: "server", Draft: &draft}
	first := apiDecode[struct {
		ID string `json:"experimentId"`
	}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "one", "", 201))
	create.Name = "Two"
	second := apiDecode[struct {
		ID string `json:"experimentId"`
	}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "two", "", 201))
	path := "/v1/eval-experiments?projectId=evaluation&limit=1"
	page := apiDecode[struct {
		Items []evalstore.PublicSummary `json:"items"`
		Page  evalPageInfo              `json:"page"`
	}](t, h.request(t, "GET", path, nil, "", "", 200))
	if len(page.Items) != 1 || page.Items[0].ID != second.ID || page.Page.NextCursor == nil {
		t.Fatal("experiment sort is not updated-time descending")
	}
	nextPath := path + "&cursor=" + url.QueryEscape(*page.Page.NextCursor)
	next := apiDecode[struct {
		Items []evalstore.PublicSummary `json:"items"`
	}](t, h.request(t, "GET", nextPath, nil, "", "", 200))
	if len(next.Items) != 1 || next.Items[0].ID != first.ID {
		t.Fatal("keyset skipped a row")
	}
	h.request(t, "GET", "/v1/eval-experiments?projectId=foreign&limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor), nil, "", "", 422)
	h.request(t, "PATCH", "/v1/eval-experiments/"+first.ID, evaldomain.DraftUpdate{Name: "Changed", Draft: draft}, "edit", `"1"`, 200)
	h.request(t, "GET", nextPath, nil, "", "", 409)
	for _, query := range []string{"limit=101", "limit=0", "limit=-1", "limit=1&limit=2", "unknown=x", "state=garbage", "cursor=untrusted"} {
		h.request(t, "GET", "/v1/eval-experiments?"+query, nil, "", "", 422)
	}
	h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", data, "second-import", "", 201)
	datasetPath := "/v1/projects/evaluation/eval-datasets?limit=1"
	datasets := apiDecode[struct {
		Page evalPageInfo `json:"page"`
	}](t, h.request(t, "GET", datasetPath, nil, "", "", 200))
	if datasets.Page.NextCursor == nil {
		t.Fatal("dataset revision page missing")
	}
	h.request(t, "POST", "/v1/projects/evaluation/eval-datasets", data, "third-import", "", 201)
	h.request(t, "GET", datasetPath+"&cursor="+url.QueryEscape(*datasets.Page.NextCursor), nil, "", "", 409)
}

func TestEvalPostgresNativeAuthoringCommandsAndPublicViews(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		t.Run(kind, func(t *testing.T) {
			h := newEvalAPIHarness(t)
			h.request(t, "GET", "/v1/eval-capabilities", nil, "", "", 200)
			draft, _ := h.dataset(t, kind)
			h.request(t, "GET", "/v1/projects/evaluation/eval-datasets", nil, "", "", 200)
			casesPath := "/v1/projects/evaluation/eval-datasets/" + draft.Dataset.ID + "/revisions/" + draft.Dataset.Revision + "/cases?limit=1"
			first := apiDecode[struct {
				Items []evaldomain.Case `json:"items"`
				Page  evalPageInfo      `json:"page"`
			}](t, h.request(t, "GET", casesPath, nil, "", "", 200))
			if len(first.Items) != 1 || first.Page.NextCursor == nil {
				t.Fatal("case page missing")
			}
			h.request(t, "GET", casesPath+"&cursor="+url.QueryEscape(*first.Page.NextCursor), nil, "", "", 200)
			create := evaldomain.CreateExperiment{Name: "Browser experiment", ControlMode: "server", Draft: &draft}
			receipt := h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "create", "", 201)
			ref := apiDecode[struct {
				ID string `json:"experimentId"`
			}](t, receipt)
			e := h.get(t, ref.ID)
			if e.State != "draft" || e.Expected != 8 {
				t.Fatal("draft projection", e.State, e.Expected)
			}
			h.request(t, "GET", "/v1/eval-experiments?projectId=evaluation&limit=1", nil, "", "", 200)
			h.request(t, "PATCH", "/v1/eval-experiments/"+e.ID, evaldomain.DraftUpdate{Name: "Edited", Draft: draft}, "patch", `"1"`, 200)
			prepared := h.command(t, e.ID, "prepare")
			command := apiDecode[struct {
				ID string `json:"commandId"`
			}](t, prepared)
			h.tick(t)
			e = h.get(t, e.ID)
			if e.State != "ready" || e.PlanSHA256 == nil {
				t.Fatalf("prepare state=%s diagnostic=%s", e.State, e.Diagnostics)
			}
			h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/commands/"+command.ID, nil, "", "", 200)
			members := h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/members?limit=1", nil, "", "", 200)
			page := apiDecode[struct {
				Items []evalservice.MemberView `json:"items"`
				Page  evalPageInfo             `json:"page"`
			}](t, members)
			if len(page.Items) != 1 || page.Page.NextCursor == nil {
				t.Fatal("member page missing")
			}
			h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/members?limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor), nil, "", "", 200)
			replay := h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", create, "create", "", 201)
			if !bytes.Equal(replay.Body.Bytes(), receipt.Body.Bytes()) {
				t.Fatal("creation replay changed after Prepare")
			}
			h.request(t, "POST", "/v1/eval-experiments/"+e.ID+"/members/"+page.Items[0].Member.ID+"/submissions", evaldomain.Submission{PlanSHA256: *e.PlanSHA256}, "wrong-owner", "", 409)
			h.command(t, e.ID, "start")
			h.tick(t)
			h.tick(t)
			e = h.get(t, e.ID)
			var count int
			table := "workflow_runs"
			if kind == "audit" {
				table = "audits"
			}
			if err := h.pool.QueryRow(t.Context(), "SELECT count(*) FROM "+table).Scan(&count); err != nil || count != 1 {
				t.Fatal("ordinary execution count", count, err)
			}
			h.request(t, "GET", "/v1/eval-experiments/"+e.ID+"/members?limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor), nil, "", "", 409)
			duplicate := apiDecode[struct {
				ID string `json:"experimentId"`
			}](t, h.command(t, e.ID, "duplicate"))
			h.request(t, "DELETE", "/v1/eval-experiments/"+duplicate.ID, map[string]any{}, "delete", `"1"`, 202)
			h.tick(t)
			h.request(t, "DELETE", "/v1/eval-experiments/"+duplicate.ID, map[string]any{}, "delete", `"1"`, 202)
		})
	}
}
