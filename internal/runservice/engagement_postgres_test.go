package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

type runEngagementGuard struct{ barrier *credentials.LifecycleBarrier }

func (g runEngagementGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	return g.barrier.WithCredentialReferences(ctx, fn)
}

type runEngagementFixture struct {
	ctx         context.Context
	pool        *pgxpool.Pool
	service     *Service
	credentials *credentials.RuntimeCredentialService
	artifacts   *artifacts.Service
	source      artifacts.ScopedStore
	projectID   *string
}

func newRunEngagementFixture(t *testing.T, project bool, declaration string) *runEngagementFixture {
	t.Helper()
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	t.Cleanup(cancel)
	admin := isolatedRunServicePool(t, ctx, dsn)
	cfg := admin.Config().Copy()
	cfg.MaxConns, cfg.MinConns, cfg.MinIdleConns = 1, 0, 0
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	root := t.TempDir()
	for _, dir := range []string{"agent-templates", "workflows", "instructions", "model-policies", "llm-gateways", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, rel := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "scan-catalog", rel))
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(rel, "workflows/") {
			data = []byte(strings.Replace(string(data), "inputs: {}", declaration, 1))
		}
		if err := os.WriteFile(filepath.Join(root, rel), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	catalog, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	barrier := credentials.NewLifecycleBarrier()
	cipher, err := credentials.NewTokenCipher([]byte(strings.Repeat("p", 32)))
	if err != nil {
		t.Fatal(err)
	}
	creds, err := credentials.NewRuntimeCredentialService(credentials.RuntimeCredentialServiceOptions{
		Pool: pool, Lease: leasetest.Hold(t, admin).Lease, Cipher: cipher, Barrier: barrier,
		Usage: credentials.NewRuntimeCredentialRepository(pool), Projects: projectstore.NewPostgresStore(pool),
		Audits: hookstore.NewCredentialHolds(pool, artifactpgtest.NewStorage()),
	})
	if err != nil {
		t.Fatal(err)
	}
	as := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	source, err := as.User("policy-owner")
	if err != nil {
		t.Fatal(err)
	}
	var projectID *string
	if project {
		id := "policy-project"
		_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "policy-owner", Kind: projectstore.KindProject, Name: "Policy", IdempotencyKey: id, RequestDigest: testDigest("2")})
		if err != nil {
			t.Fatal(err)
		}
		source, err = as.Project(id)
		if err != nil {
			t.Fatal(err)
		}
		projectID = &id
	}
	service, err := New(Options{Runs: runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), Workflows: catalog,
		LLMCredentials: emptyCredentialLookup{}, CredentialGuard: runEngagementGuard{barrier}, RuntimeCredentials: creds,
		Projects: projectstore.NewPostgresStore(pool), PublicTransaction: postgresPublicTransaction(pool)})
	if err != nil {
		t.Fatal(err)
	}
	return &runEngagementFixture{ctx, pool, service, creds, as, source, projectID}
}

const runPolicyDeclaration = "inputs:\n    network_policy: {required: true, mediaTypes: [application/vnd.contractor.engagement+json]}"

func runPolicyDocument() engagementcontract.Document {
	limit := uint32(2)
	return engagementcontract.Document{Schema: engagementcontract.Schema,
		Egress:     engagementcontract.Egress{Routes: []engagementcontract.Route{{Origin: "https://target.example", PathPrefixes: []string{"/api"}, Methods: []string{"GET"}, AddressRanges: []string{"10.20.0.0/16"}}}, Exclusions: []engagementcontract.Route{}},
		Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{"https://target.example"}}},
		Limits:     &engagementcontract.HTTPLimits{TargetRequests: &limit}}
}
func (f *runEngagementFixture) addCredential(t *testing.T, doc *engagementcontract.Document, id, owner string, kind runtimesettings.RuntimeCredentialKind) {
	t.Helper()
	var material credentials.RuntimeCredentialMaterial
	var err error
	if kind == runtimesettings.RuntimeCredentialTargetBearer {
		material, err = credentials.NewTargetBearerCredential("run-policy-private-canary")
	} else {
		material, err = credentials.NewTargetBasicCredential("test", "run-policy-private-canary")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer material.Destroy()
	if _, err := f.credentials.Create(f.ctx, credentials.RuntimeCredentialCreateRequest{CredentialID: id, ActorID: owner, IdempotencyKey: id, Material: material}); err != nil {
		t.Fatal(err)
	}
	doc.Identities = append(doc.Identities, engagementcontract.Identity{ID: id, Kind: "bearer", Origins: []string{"https://target.example"}, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: id, Kind: runtimesettings.RuntimeCredentialTargetBearer}})
}
func (f *runEngagementFixture) params(t *testing.T, id string, document engagementcontract.Document) PublicCreateParams {
	t.Helper()
	body, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	exact := writeExact(t, f.ctx, f.source, "policies", id, engagementcontract.MediaType, body)
	params := PublicCreateParams{OwnerID: "policy-owner", ProjectID: f.projectID, Workflow: "nuclei-target@1", Parameters: map[string]string{"target": "https://target.example/api"}, Inputs: map[string]contracts.ArtifactRef{"network_policy": exact.Ref}, IdempotencyKey: id, NewRunID: func() (string, error) { return id, nil }}
	params.RequestDigest, err = PublicRequestDigest(params)
	if err != nil {
		t.Fatal(err)
	}
	return params
}
func (f *runEngagementFixture) stage(t *testing.T, run runstore.WorkflowRun) string {
	t.Helper()
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	if _, err := runs.TransitionRun(f.ctx, run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	stage, err := runs.CreateStageExecution(f.ctx, runstore.CreateStageExecutionParams{StageExecutionID: "stage-" + run.RunID, RunID: run.RunID,
		StageName: workflow.EntryStage, Attempt: 1, LaneClass: resolved.LaneClassTool,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: body, StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}}})
	if err != nil {
		t.Fatal(err)
	}
	return stage.StageExecutionID
}

func TestPostgresRunEngagementPinsOriginalInputAndCredentials(t *testing.T) {
	for _, project := range []bool{false, true} {
		t.Run(fmt.Sprint("project=", project), func(t *testing.T) {
			f := newRunEngagementFixture(t, project, runPolicyDeclaration)
			doc := runPolicyDocument()
			f.addCredential(t, &doc, "account-a", "policy-owner", runtimesettings.RuntimeCredentialTargetBearer)
			if project {
				material, err := credentials.NewHTTPOriginBearerCredential("unused-project-canary")
				if err != nil {
					t.Fatal(err)
				}
				defer material.Destroy()
				if _, err := f.credentials.Create(f.ctx, credentials.RuntimeCredentialCreateRequest{CredentialID: "unavailable-default", ActorID: "other-owner", IdempotencyKey: "project-default", Material: material}); err != nil {
					t.Fatal(err)
				}
				projects := projectstore.NewPostgresStore(f.pool)
				current, err := projects.Get(f.ctx, "policy-owner", *f.projectID)
				if err != nil {
					t.Fatal(err)
				}
				_, err = projects.Update(f.ctx, projectstore.UpdateParams{OwnerID: current.OwnerID,
					ProjectID: current.ProjectID, ExpectedRevision: current.Revision, Name: current.Name,
					HTTPTarget: &runtimesettings.HTTPOriginTargetRef{URL: "https://default.example/",
						Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "unavailable-default", Kind: runtimesettings.RuntimeCredentialOriginBearer}}})
				if err != nil {
					t.Fatal(err)
				}
			}
			params := f.params(t, "policy-run", doc)
			// A versionless request resolves once, then pins that exact source.
			source := params.Inputs["network_policy"]
			versionless := source
			versionless.Revision = nil
			params.Inputs["network_policy"] = versionless
			params.RequestDigest, _ = PublicRequestDigest(params)
			created, err := f.service.CreatePublic(f.ctx, params)
			if err != nil || !created.Created {
				t.Fatalf("create: %v", err)
			}
			if !slices.Equal(created.Run.EngagementCredentialIDs, []string{"account-a"}) || created.Run.ProjectHTTPTarget != nil {
				t.Fatal("Run lost credentials or inherited a default target")
			}
			if _, err := f.credentials.Delete(f.ctx, "account-a", "policy-owner"); !errors.Is(err, credentials.ErrRuntimeCredentialInUse) {
				t.Fatalf("active credential hold: %v", err)
			}
			stage := f.stage(t, created.Run)
			pin, ordinary, err := ReadAllocationEngagement(f.ctx, f.pool, created.Run.RunID, stage, artifactpgtest.NewStorage())
			if err != nil || !ordinary || pin == nil || pin.AuthorizedBy != "policy-owner" || !pin.AuthorizedAt.Equal(created.Run.CreatedAt) {
				t.Fatalf("allocation: %v", err)
			}
			if _, _, err := ReadAllocationEngagement(f.ctx, f.pool, "foreign-run", stage, artifactpgtest.NewStorage()); !errors.Is(err, runstore.ErrNotFound) {
				t.Fatalf("foreign Stage: %v", err)
			}
			before, _ := json.Marshal(pin)
			doc.Egress.Routes[0].PathPrefixes = []string{"/"}
			body, _ := json.Marshal(doc)
			if _, err := f.source.Write(f.ctx, versionless, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: body}, source.Revision); err != nil {
				t.Fatal(err)
			}
			runArtifacts, _ := f.artifacts.Run(created.Run.RunID)
			if _, err := runArtifacts.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "network_policy"}, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: body}, pin.Artifact.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			again, _, err := ReadAllocationEngagement(f.ctx, f.pool, created.Run.RunID, stage, artifactpgtest.NewStorage())
			if err != nil {
				t.Fatal(err)
			}
			after, _ := json.Marshal(again)
			if string(before) != string(after) {
				t.Fatal("mutable policy changed allocation authority")
			}
			f.service.publicTransaction = func(context.Context, func(PublicRunWriter, *artifacts.Service, RuntimeCredentialValidator) error) error {
				t.Fatal("replay resolved mutable dependencies")
				return nil
			}
			replay, err := f.service.CreatePublic(f.ctx, params)
			if err != nil || !replay.Replayed || replay.Run.RunID != created.Run.RunID {
				t.Fatalf("replay: %v", err)
			}
		})
	}
}

func TestPostgresRunEngagementRejectsInvalidAdmissionAtomically(t *testing.T) {
	cases := []string{"foreign-credential", "wrong-kind", "missing-credential", "invalid-policy", "oversized", "ambiguous-slots", "mixed-media", "implicit-slot"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			declaration := runPolicyDeclaration
			if name == "ambiguous-slots" {
				declaration += "\n    extra: {required: false, mediaTypes: [application/vnd.contractor.engagement+json]}"
			}
			if name == "mixed-media" {
				declaration = strings.Replace(declaration, "+json]", "+json, text/plain]", 1)
			}
			if name == "implicit-slot" {
				declaration = strings.Replace(declaration, engagementcontract.MediaType, "\"*/*\"", 1)
			}
			f := newRunEngagementFixture(t, false, declaration)
			doc := runPolicyDocument()
			if name == "foreign-credential" {
				f.addCredential(t, &doc, "foreign", "other-owner", runtimesettings.RuntimeCredentialTargetBearer)
			}
			if name == "wrong-kind" {
				f.addCredential(t, &doc, "wrong", "policy-owner", runtimesettings.RuntimeCredentialTargetBasic)
			}
			if name == "missing-credential" {
				doc.Identities = append(doc.Identities, engagementcontract.Identity{ID: "missing", Kind: "bearer", Origins: doc.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "missing", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
			}
			if name == "invalid-policy" {
				doc.Egress.Routes[0].Methods = []string{}
			}
			params := f.params(t, "rejected-run", doc)
			if name == "oversized" {
				large := writeExact(t, f.ctx, f.source, "policies", "large", engagementcontract.MediaType, []byte(strings.Repeat(" ", engagementcontract.MaxDocumentBytes+1)))
				params.Inputs["network_policy"] = large.Ref
				params.RequestDigest, _ = PublicRequestDigest(params)
			}
			_, err := f.service.CreatePublic(f.ctx, params)
			want := ErrInvalid
			if strings.Contains(name, "credential") || name == "wrong-kind" {
				want = credentials.ErrRuntimeCredentialNotFound
			}
			if !errors.Is(err, want) {
				t.Fatalf("admission error = %v, want %v", err, want)
			}
			var rows int
			if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM workflow_runs`).Scan(&rows); err != nil || rows != 0 {
				t.Fatalf("rejected Run survived: %d %v", rows, err)
			}
		})
	}
}

func TestPostgresRunEngagementCredentialDeleteRaceAndDraftLifetime(t *testing.T) {
	f := newRunEngagementFixture(t, false, runPolicyDeclaration)
	doc := runPolicyDocument()
	f.addCredential(t, &doc, "race-account", "policy-owner", runtimesettings.RuntimeCredentialTargetBearer)
	params := f.params(t, "race-run", doc)
	entered, proceed := make(chan struct{}), make(chan struct{})
	original := f.service.publicTransaction
	f.service.publicTransaction = func(ctx context.Context, fn func(PublicRunWriter, *artifacts.Service, RuntimeCredentialValidator) error) error {
		return original(ctx, func(runs PublicRunWriter, as *artifacts.Service, validator RuntimeCredentialValidator) error {
			close(entered)
			select {
			case <-proceed:
			case <-ctx.Done():
				return ctx.Err()
			}
			return fn(runs, as, validator)
		})
	}
	created, deleted := make(chan error, 1), make(chan error, 1)
	go func() { _, err := f.service.CreatePublic(f.ctx, params); created <- err }()
	select {
	case <-entered:
	case <-f.ctx.Done():
		t.Fatal("creation did not enter guarded transaction")
	}
	go func() { _, err := f.credentials.Delete(f.ctx, "race-account", "policy-owner"); deleted <- err }()
	close(proceed)
	if err := <-created; err != nil {
		t.Fatal(err)
	}
	if err := <-deleted; !errors.Is(err, credentials.ErrRuntimeCredentialInUse) {
		t.Fatalf("deletion passed committed Run reference: %v", err)
	}
	f.service.publicTransaction = original
	draftDoc := runPolicyDocument()
	f.addCredential(t, &draftDoc, "draft-account", "policy-owner", runtimesettings.RuntimeCredentialTargetBearer)
	draftParams := f.params(t, "draft-run", draftDoc)
	if _, err := f.credentials.Delete(f.ctx, "draft-account", "policy-owner"); err != nil {
		t.Fatalf("a document alone held its credential: %v", err)
	}
	if _, err := f.service.CreatePublic(f.ctx, draftParams); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("Run accepted a deleted draft credential: %v", err)
	}
}

func TestPostgresRunEngagementHoldsCredentialUntilAllocationRelease(t *testing.T) {
	f := newRunEngagementFixture(t, false, runPolicyDeclaration)
	doc := runPolicyDocument()
	f.addCredential(t, &doc, "held-account", "policy-owner", runtimesettings.RuntimeCredentialTargetBearer)
	created, err := f.service.CreatePublic(f.ctx, f.params(t, "release-run", doc))
	if err != nil {
		t.Fatal(err)
	}
	stageID := f.stage(t, created.Run)
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(created.Run.WorkflowSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	agent := workflow.Stages[workflow.EntryStage].Agents["scanner"].Template
	provenance := runtimesettings.ResolvedRuntimeConfigProvenance{
		Default: runtimesettings.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1,
			Config: runtimesettings.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}},
		RunLabels: []runtimesettings.RuntimeLabelBindingProvenance{}, AgentLabels: []runtimesettings.RuntimeLabelBindingProvenance{},
		RuntimeAdapters: []runtimesettings.RuntimeAdapterRef{}, RuntimeCredentialRefs: []runtimesettings.RuntimeCredentialRef{},
	}
	if err := runs.RecordStageAllocation(f.ctx, runstore.StageAllocation{
		AllocationID: "held-allocation", StageExecutionID: stageID, LogicalAgentName: "scanner", Namespace: "scanner",
		AgentTemplateRef: agent.Ref, WorkerRuntimeRef: agent.Runtime, RuntimeAgentID: strings.Repeat("a", 64),
		RuntimeAgentInstanceID: "held-instance", RuntimeAgentLabelRevision: 1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              &runstore.AllocationRuntimeConfiguration{Provenance: provenance},
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	if err := runs.EnterAborting(f.ctx, runstore.EnterAbortingParams{StageExecutionID: stageID, ExpectedState: runstore.StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion, Termination: runstore.StageTermination{Outcome: runstore.TerminationInterrupted,
			Code: "runtime_unavailable", Message: "fixture", Retryable: true, Phase: runstore.TerminationPreparing, OccurredAt: now},
		AbortID: "held-abort", Deadline: now.Add(time.Minute), Reason: runstore.Reason{Code: "stage_interrupted"}}); err != nil {
		t.Fatal(err)
	}
	if err := runs.CompleteStageTermination(f.ctx, stageID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.RecordStageTransitionDecision(f.ctx, runstore.RecordStageTransitionDecisionParams{SourceExecutionID: stageID, RunID: created.Run.RunID, Action: runstore.StageTransitionFail}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(f.ctx, created.Run.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "stage_failed"}); err != nil {
		t.Fatal(err)
	}
	_, err = f.credentials.Delete(f.ctx, "held-account", "policy-owner")
	var inUse *credentials.RuntimeCredentialInUseError
	if !errors.As(err, &inUse) || !slices.Equal(inUse.Usage.AllocationIDs, []string{"held-allocation"}) || len(inUse.Usage.RunIDs) != 0 {
		t.Fatalf("terminal unreleased credential: %v", err)
	}
	if err := runs.MarkStageAllocationReleased(f.ctx, "held-allocation"); err != nil {
		t.Fatal(err)
	}
	if _, err := f.credentials.Delete(f.ctx, "held-account", "policy-owner"); err != nil {
		t.Fatalf("released credential: %v", err)
	}
}
