package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Run creation and Audit start measure only the stored size of a Skill
// package, so the bytes need not form one. An oversized package exceeds the
// stored package bound (LIM-SKL-17); the selected Skills, each of
// sharedSkillBytes within that bound, together exceed the per-AgentTemplate
// stored-byte limit (LIM-SKL-22).
const (
	oversizedSkillBytes = agentskills.MaximumArchiveBytes + 1
	sharedSkillBytes    = agentskills.MaximumTemplateStoredBytes/5 + 1
)

// selectedSkills are the Skills one AgentTemplate selects in these tests.
var selectedSkills = []string{"alpha", "beta", "delta", "epsilon", "gamma"}

// selectSkills makes the AgentTemplate at path select the Skills names,
// inserting the selection before anchor.
func selectSkills(t *testing.T, path, anchor string, names ...string) {
	t.Helper()
	template, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	refs := make([]string, 0, len(names))
	for _, name := range names {
		refs = append(refs, "{namespace: skills, name: "+name+"}")
	}
	edited := bytes.Replace(template, []byte(anchor), []byte("  skills: ["+strings.Join(refs, ", ")+"]\n"+anchor), 1)
	if bytes.Equal(edited, template) {
		t.Fatalf("%s has no %q to select Skills before", path, anchor)
	}
	if err := os.WriteFile(path, edited, 0o600); err != nil {
		t.Fatal(err)
	}
}

// writeSkills writes size bytes as the current package of every Skill name
// in store, advancing a binding that exists.
func writeSkills(t *testing.T, ctx context.Context, store artifacts.ScopedStore, size int, names ...string) {
	t.Helper()
	for _, name := range names {
		ref := artifacts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: name}
		var current *string
		if metadata, err := store.Metadata(ctx, ref); err == nil {
			current = metadata.Ref.Revision
		} else if !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatal(err)
		}
		payload := artifacts.Payload{MediaType: agentskills.MediaType, Data: bytes.Repeat([]byte{'s'}, size)}
		if _, err := store.Write(ctx, ref, payload, current); err != nil {
			t.Fatalf("write skills/%s: %v", name, err)
		}
	}
}

// checkSkillFailure checks that response is the public error envelope of
// code with its registered status and retryability, and names no internal
// cause.
func checkSkillFailure(t *testing.T, response *httptest.ResponseRecorder, status int, code string, retryable bool) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Errorf("decode error response %q: %v", response.Body.String(), err)
		return
	}
	if response.Code != status || envelope.Code != code || envelope.Retryable != retryable ||
		strings.Contains(envelope.Message, skillOutageCanary) {
		t.Errorf("Skill failure = %d %+v, want %d %s retryable=%t", response.Code, envelope, status, code, retryable)
	}
}

const skillOutageCanary = "injected Skill source read outage"

// skillSourceOutage fails every Skill source read while outage is set, as a
// database failure inside the Run-creation transaction would; every other
// read and write reaches PostgreSQL.
type skillSourceOutage struct {
	artifacts.Repository
	outage *atomic.Bool
}

func (r skillSourceOutage) Metadata(
	ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
) (artifacts.Metadata, error) {
	if r.outage.Load() && ref.Namespace == contracts.AgentSkillNamespace {
		return artifacts.Metadata{}, errors.New(skillOutageCanary)
	}
	return r.Repository.Metadata(ctx, scope, ref)
}

func countWorkflowRuns(t *testing.T, ctx context.Context, pool *pgxpool.Pool) int {
	t.Helper()
	var stored int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	return stored
}

// TestPostgresRunCreationAnswersSkillFailuresWithTheirCodes drives POST
// /v1/runs and POST /v1/projects/{projectId}/runs on the real stores into
// every Skill failure Run creation can meet and checks each answer against
// the public contract (SKL-R117): a missing current Skill is a committed
// missing-source result, not a failure (SKL-R69); a transient source read is
// the retryable 503 skill_artifact_unavailable that commits nothing (ERR-R8),
// after which the same request succeeds; a package above the stored package
// bound and Skills above the per-AgentTemplate stored-byte limit are the
// permanent 422 skill_limit_exceeded. None of them is internal_error.
func TestPostgresRunCreationAnswersSkillFailuresWithTheirCodes(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	root := copyPublicConfigTree(t)
	selectSkills(t, filepath.Join(root, "agent-templates", "artifact_builder.yaml"),
		"  sandboxProfile: local-workdir@1\n", selectedSkills...)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projects := projectstore.NewPostgresStore(pool)
	details, err := runservice.NewRunDetailReader(runservice.PostgresRunDetailTransaction(pool, artifactpgtest.NewStorage(), nil))
	if err != nil {
		t.Fatal(err)
	}
	outage := &atomic.Bool{}
	var runIDs atomic.Int64
	fixture := newHandlerFixtureWithAuth(t, root, newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Runs, d.Artifacts, d.Projects, d.RunDetails = runs, artifactService, projects, details
		d.RunQueue, d.RunLifecycle = runs, runs
		d.NewID = func(prefix string) (string, error) { return fmt.Sprintf("%s%d", prefix, runIDs.Add(1)), nil }
		d.RunCreator = newTestRunCreator(t, *d, runs, integrationUnitOfWork{
			pool: pool, credentials: d.Credentials,
			wrapArtifacts: func(repository artifacts.Repository) artifacts.Repository {
				return skillSourceOutage{Repository: repository, outage: outage}
			},
		}, true)
	})
	user, err := artifactService.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	source, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("Skill Run source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-skills", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Skills", IdempotencyKey: "project-skills", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	router := publicContractRouter(t)
	body, err := json.Marshal(map[string]any{
		"workflow": "artifact-copy@1", "parameters": map[string]string{},
		"artifacts": map[string]contracts.ArtifactRef{"source": source.Ref},
	})
	if err != nil {
		t.Fatal(err)
	}
	create := func(path, key string) *httptest.ResponseRecorder {
		t.Helper()
		request := jsonContractRequest(http.MethodPost, path, string(body), idempotencyKeyHeader, key)
		return serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	}
	accepted := func(response *httptest.ResponseRecorder) runstore.WorkflowRun {
		t.Helper()
		var created createRunResponse
		if response.Code != http.StatusAccepted || json.Unmarshal(response.Body.Bytes(), &created) != nil {
			t.Fatalf("create Skill Run = %d %s", response.Code, response.Body.String())
		}
		run, err := runs.GetRun(ctx, created.RunID)
		if err != nil {
			t.Fatal(err)
		}
		return run
	}

	// The owner holds neither Skill: Run creation commits the missing-source
	// results, and the Run's initialization fails later (SKL-R69).
	missing := accepted(create("/v1/runs", "skills-missing"))
	if missing.State != runstore.RunInitializing || len(missing.SkillSnapshot) != len(selectedSkills) {
		t.Fatalf("Run with missing Skills = %+v", missing)
	}
	for _, skill := range missing.SkillSnapshot {
		if skill.Source != nil {
			t.Fatalf("missing Skill %s selected %+v", skill.Name, skill.Source)
		}
	}

	// A Skill source read fails: nothing is committed and the same request
	// succeeds once the store answers.
	writeSkills(t, ctx, user, 1024, selectedSkills...)
	outage.Store(true)
	checkSkillFailure(t, create("/v1/runs", "skills-outage"), http.StatusServiceUnavailable,
		agentskills.CodeArtifactUnavailable, true)
	if stored := countWorkflowRuns(t, ctx, pool); stored != 1 {
		t.Errorf("a failed Skill selection stored %d Runs, want 1", stored)
	}
	outage.Store(false)
	selected := accepted(create("/v1/runs", "skills-outage"))
	for _, skill := range selected.SkillSnapshot {
		if skill.Source == nil {
			t.Fatalf("retried Skill selection = %+v", selected.SkillSnapshot)
		}
	}

	// One package above the stored package bound, then packages within it
	// that together exceed the AgentTemplate's stored-byte limit, on either
	// Run creation route.
	writeSkills(t, ctx, user, oversizedSkillBytes, "alpha")
	checkSkillFailure(t, create("/v1/runs", "skills-oversized"), http.StatusUnprocessableEntity,
		agentskills.CodeLimitExceeded, false)
	writeSkills(t, ctx, user, sharedSkillBytes, selectedSkills...)
	checkSkillFailure(t, create("/v1/runs", "skills-over-template"), http.StatusUnprocessableEntity,
		agentskills.CodeLimitExceeded, false)
	checkSkillFailure(t, create("/v1/projects/project-skills/runs", "skills-over-template-project"),
		http.StatusUnprocessableEntity, agentskills.CodeLimitExceeded, false)
	if stored := countWorkflowRuns(t, ctx, pool); stored != 2 {
		t.Errorf("rejected Skill selections stored %d Runs, want 2", stored)
	}
}

// TestPostgresAuditStartAnswersSkillFailuresWithTheirCodes starts an Audit
// whose child Workflow selects Skills on the real stores and checks each
// answer against the public contract (SKL-R117): Audit start pins the Skill
// packages for later child Runs, so a missing current Skill is the permanent
// 422 skill_artifact_not_found, and a package above the stored package bound
// or Skills above the per-AgentTemplate stored-byte limit are the permanent
// 422 skill_limit_exceeded. None changes the draft, and the same start
// succeeds once the owner's packages fit.
func TestPostgresAuditStartAnswersSkillFailuresWithTheirCodes(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	configuration := loadPublicAuditConfigurationWith(t, func(root string) {
		selectSkills(t, filepath.Join(root, "agent-templates", "workspace_source_graph_analyst_v3_memory.yaml"),
			"  sandboxProfile: local-workdir@1\n", selectedSkills...)
	})
	gateway, err := configuration.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &scriptedLLMCredentialLookup{metadata: map[string]resolved.CredentialMetadata{
		"development-worker": {
			Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
	}}
	audits, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(),
		Pool: pool, Profiles: configuration,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil },
		),
		CredentialGuard: newFakeManagedCredentials(),
		Now:             func() time.Time { return time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-skills", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Audit Skills", IdempotencyKey: "create-audit-skills-project",
		RequestDigest: auditHandlerDigest("audit-skills-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("fixture source archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	checklist, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist"},
		artifacts.Payload{MediaType: "application/json", Data: []byte(
			`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-auth","version":"1","statement":"Verify authentication boundaries.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`,
		)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := audits.CreateDraft(ctx, auditservice.CreateDraftParams{
		AuditID: "audit-skills", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        auditservice.ProfileSelector{Name: "public-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"source": source.Ref, "checklist": checklist.Ref},
		RuntimeLabels:  []string{},
		Scope:          auditservice.Scope{Objective: "Review authentication"},
		IdempotencyKey: "create-audit-skills", RequestDigest: auditHandlerDigest("create-audit-skills"),
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(d *testServices) { d.Audits, d.Projects = audits, projects })
	router := publicContractRouter(t)
	start := func() *httptest.ResponseRecorder {
		t.Helper()
		request := jsonContractRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", "",
			idempotencyKeyHeader, "start-audit-skills", "If-Match", fmt.Sprintf("%q", fmt.Sprint(draft.Revision)))
		return serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	}
	unchanged := func() {
		t.Helper()
		current, err := audits.Get(ctx, project.OwnerID, draft.AuditID)
		if err != nil || current.State != auditstore.AuditDraft || current.Revision != draft.Revision {
			t.Errorf("failed start changed the draft = (%+v, %v)", current, err)
		}
	}
	user, err := artifactService.User(project.OwnerID)
	if err != nil {
		t.Fatal(err)
	}

	checkSkillFailure(t, start(), http.StatusUnprocessableEntity, agentskills.CodeArtifactNotFound, false)
	unchanged()

	writeSkills(t, ctx, user, 1024, selectedSkills...)
	writeSkills(t, ctx, user, oversizedSkillBytes, "alpha")
	checkSkillFailure(t, start(), http.StatusUnprocessableEntity, agentskills.CodeLimitExceeded, false)
	unchanged()

	writeSkills(t, ctx, user, sharedSkillBytes, selectedSkills...)
	checkSkillFailure(t, start(), http.StatusUnprocessableEntity, agentskills.CodeLimitExceeded, false)
	unchanged()

	writeSkills(t, ctx, user, 1024, selectedSkills...)
	response := start()
	var started auditStartResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &started) != nil ||
		started.Audit.State != auditstore.AuditActive {
		t.Fatalf("start with fitting Skills = %d %s", response.Code, response.Body.String())
	}
}
