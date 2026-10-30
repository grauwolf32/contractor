package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// auditListTrace counts the statements of a traced pool. Before the first
// statement whose text contains marker it runs interleave once.
type auditListTrace struct {
	mu          sync.Mutex
	statements  int
	marker      string
	interleave  func(*pgx.Conn) error
	interleaved bool
	err         error
}

func (trace *auditListTrace) TraceQueryStart(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	trace.mu.Lock()
	trace.statements++
	interleave := trace.interleave
	if interleave == nil || !strings.Contains(data.SQL, trace.marker) {
		trace.mu.Unlock()
		return ctx
	}
	trace.interleave = nil
	trace.mu.Unlock()
	err := interleave(conn)
	trace.mu.Lock()
	trace.interleaved, trace.err = true, err
	trace.mu.Unlock()
	return ctx
}

func (*auditListTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns the number of statements since the last take.
func (trace *auditListTrace) take() int {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	count := trace.statements
	trace.statements = 0
	return count
}

// preparedListFixture seeds Audits that pin a two-role preparation profile
// and drives their prepare attempts through the Audit store, Run and
// artifact steps the Audit Controller uses.
type preparedListFixture struct {
	t         *testing.T
	ctx       context.Context
	pool      *pgxpool.Pool
	store     *auditstoretest.Store
	artifacts *artifacts.Service
	catalog   auditservice.ProfileCatalog
	profile   auditprofile.ResolvedAuditProfile
	snapshot  json.RawMessage
	selection json.RawMessage
	baseline  json.RawMessage
	inputs    map[string]auditstore.ExactArtifact
	owner     string
	claims    map[string]auditstore.ControllerClaim
	accepted  map[string]map[string]auditstore.PreparationOutput
}

func newPreparedListFixture(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner string, projectIDs ...string) *preparedListFixture {
	t.Helper()
	root := configtest.CopyWithPolicies(t, "../../auditservice/testdata/catalog")
	data := contractstest.ReadFile(t, "api", "testdata", "audit-composition", "prepared-openapi-scan.yaml")
	data = []byte(strings.Replace(string(data), "    scan:\n", `    refine-api:
      kind: prepare
      ref: openapi-from-workspace@7
      maxRunAttempts: 2
      inputs:
        source: {source: audit-input, name: source}
        existing_openapi: {source: prepare-output, role: generate-api, name: api}
      parameters: {}
      outputs: {api: openapi, validation: openapi_validation_report}
    scan:
`, 1))
	if err := os.WriteFile(filepath.Join(root, "audit-profiles", "prepared.yaml"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, err := configload.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	inputs := map[string]auditstore.ExactArtifact{}
	for index, projectID := range projectIDs {
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: projectID,
			IdempotencyKey: projectID, RequestDigest: auditHandlerDigest(projectID),
		}); err != nil {
			t.Fatal(err)
		}
		if index != 0 {
			continue
		}
		// The baseline names the first Project's inputs; the Audit store
		// pins them without resolving their Project.
		projectArtifacts, err := service.Project(projectID)
		if err != nil {
			t.Fatal(err)
		}
		for name, media := range map[string]string{"source": "application/zip", "settings": "application/json"} {
			payload := []byte(name)
			written, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name},
				artifacts.Payload{MediaType: media, Data: payload}, nil)
			if err != nil {
				t.Fatal(err)
			}
			inputs[name] = auditstore.ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: media, SizeBytes: int64(len(payload))}
		}
	}
	selection, err := json.Marshal(auditservice.DraftSelection{Labels: map[string]string{}, RequiredPlacementLabels: []string{},
		Schema: auditservice.DraftSelectionSchema, Inputs: inputs, RuntimeLabels: []string{}})
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := auditservice.EncodeBaseline(auditservice.BaselineSnapshot{Labels: map[string]string{}, RequiredPlacementLabels: []string{},
		ActiveChecks: contracts.ActiveChecksAutomatic, Schema: auditservice.BaselineSchema, Inputs: inputs, RuntimeLabels: []string{},
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), Skills: []agentskills.RunSkillSnapshot{},
		LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if err != nil {
		t.Fatal(err)
	}
	return &preparedListFixture{t: t, ctx: ctx, pool: pool,
		store:   auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch),
		catalog: catalog, artifacts: service, profile: profile, snapshot: snapshot, selection: selection,
		baseline: baseline, inputs: inputs, owner: owner, claims: map[string]auditstore.ControllerClaim{},
		accepted: map[string]map[string]auditstore.PreparationOutput{}}
}

func (f *preparedListFixture) create(projectID, auditID string) auditstore.Audit {
	f.t.Helper()
	audit, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: f.owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: f.profile.Ref.Name, Version: f.profile.Ref.Version, Digest: f.profile.Ref.Digest},
		ProfileSnapshot: f.snapshot, InputSelection: f.selection,
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 16, MaxItemsTotal: 16,
			MaxSubmittedRuns: 50, MaxItemRunAttempts: 3, MaxEvidenceBytes: 64 << 20},
		IdempotencyKey: "create-" + auditID, RequestDigest: auditHandlerDigest("create-" + auditID),
	})
	if err != nil {
		f.t.Fatal(err)
	}
	return audit
}

func (f *preparedListFixture) start(projectID, auditID string) auditstore.Audit {
	f.t.Helper()
	draft := f.create(projectID, auditID)
	started, _, err := f.store.StartPreparation(f.ctx, auditstore.StartPreparationParams{
		OwnerID: f.owner, AuditID: auditID, ExpectedRevision: draft.Revision, BaselineSnapshot: f.baseline,
		DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start-" + auditID, RequestDigest: auditHandlerDigest("start-" + auditID),
	})
	if err != nil {
		f.t.Fatal(err)
	}
	return started
}

func (f *preparedListFixture) claim(count int) {
	f.t.Helper()
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: "list-controller", Lease: 5 * time.Minute, Limit: count})
	if err != nil || len(claims) != count {
		f.t.Fatalf("claim %d Audits = (%+v, %v)", count, claims, err)
	}
	for _, claim := range claims {
		f.claims[claim.AuditID] = claim
	}
}

// attempt drives one prepare attempt to outcome: intent, submitted, failed or
// succeeded. A succeeded attempt is accepted with both mapped outputs.
func (f *preparedListFixture) attempt(projectID, auditID, role string, number int, outcome string) {
	f.t.Helper()
	ctx, claim := f.ctx, f.claims[auditID]
	execution, err := f.intent(auditID, role, number)
	if err != nil {
		f.t.Fatal(err)
	}
	if outcome == "intent" {
		return
	}
	executionID := execution.ExecutionID
	runID := "run-" + executionID
	binding := f.profile.Workflows[role]
	workflow, err := json.Marshal(binding.Workflow)
	if err != nil {
		f.t.Fatal(err)
	}
	if err := persistencepostgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := runstore.NewPostgresStore(tx).CreateAuditRun(ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: f.owner, ProjectID: &projectID,
				WorkflowName: binding.Workflow.Ref.Name, WorkflowVersion: binding.Workflow.Ref.Version, WorkflowSchemaVersion: "contractor/v1alpha1",
				WorkflowSnapshot: workflow, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		_, err := auditstoretest.NewStore(tx).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch).BindRun(ctx,
			auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return err
	}); err != nil {
		f.t.Fatal(err)
	}
	if outcome == "submitted" {
		return
	}
	f.settle(projectID, auditID, role, number, outcome)
}

// settle ends the submitted prepare attempt number of the role of the claimed
// Audit with outcome, failed or succeeded, and collects it.
func (f *preparedListFixture) settle(projectID, auditID, role string, number int, outcome string) {
	f.t.Helper()
	ctx, claim := f.ctx, f.claims[auditID]
	executionID := auditdomain.DeterministicID("prepare", auditID, role, strconv.Itoa(number))
	runID := "run-" + executionID
	var outputs []auditstore.PreparationOutput
	if outcome == "succeeded" {
		run, err := f.artifacts.Run(runID)
		if err != nil {
			f.t.Fatal(err)
		}
		for _, names := range [][3]string{{"api", "openapi", "application/yaml"}, {"validation", "openapi_validation_report", "text/markdown"}} {
			payload := []byte(`{"name":"` + names[0] + `","audit":"` + auditID + `"}`)
			written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "stage", Name: names[1]}, artifacts.Payload{MediaType: names[2], Data: payload}, nil)
			if err != nil {
				f.t.Fatal(err)
			}
			bound, err := artifactstest.BindOutput(ctx, f.artifacts, runID, names[1], written.Ref)
			if err != nil {
				f.t.Fatal(err)
			}
			outputs = append(outputs, auditstore.PreparationOutput{LogicalName: names[0], WorkflowOutput: names[1],
				Source: auditstore.ExactArtifact{Ref: bound.TargetRef, Digest: auditdomain.DigestBytes(payload), MediaType: names[2], SizeBytes: int64(len(payload))}})
		}
		if err := artifactstest.FreezeOutputs(ctx, f.artifacts, runID); err != nil {
			f.t.Fatal(err)
		}
		source, _ := artifacts.RunScope(runID)
		target, _ := artifacts.ProjectScope(projectID)
		for index := range outputs {
			retained, err := f.artifacts.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditImport, Source: source,
				SourceRef: outputs[index].Source.Ref, Target: target, Frozen: true,
				TargetRef: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(auditID), Name: executionID + "-" + outputs[index].LogicalName}})
			if err != nil {
				f.t.Fatal(err)
			}
			outputs[index].Retained = outputs[index].Source
			outputs[index].Retained.Ref = retained.TargetRef
		}
	}
	if _, err := f.pool.Exec(ctx, `UPDATE workflow_runs SET state = $2, state_reason_code = 'test-terminal',
       started_at = clock_timestamp(), finished_at = clock_timestamp() WHERE run_id = $1`, runID, outcome); err != nil {
		f.t.Fatal(err)
	}
	var generation string
	var sequence uint64
	if err := f.pool.QueryRow(ctx, `SELECT run_event_generation, next_run_event_sequence - 1 FROM workflow_runs WHERE run_id = $1`, runID).
		Scan(&generation, &sequence); err != nil {
		f.t.Fatal(err)
	}
	if _, err := f.store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: executionID,
		RunID: runID, Generation: generation, Sequence: sequence}); err != nil {
		f.t.Fatal(err)
	}
	collect := auditstore.CollectParams{Claim: claim, ExecutionID: executionID, ReceiptID: "receipt-" + executionID,
		RequestDigest: auditHandlerDigest("receipt-" + executionID), Disposition: auditstore.CollectionExecutionFailed}
	if len(outputs) != 0 {
		collect.Disposition, collect.SourceOutput, collect.PreparationOutputs = auditstore.CollectionAccepted, &outputs[0].Source, outputs
	}
	if _, _, err := f.store.Collect(ctx, collect); err != nil {
		f.t.Fatal(err)
	}
	if len(outputs) != 0 && role == "generate-api" {
		f.accepted[auditID] = map[string]auditstore.PreparationOutput{}
		for _, output := range outputs {
			f.accepted[auditID][output.LogicalName] = output
		}
	}
}

// intent records the intent of prepare attempt number of the role of the
// claimed Audit.
func (f *preparedListFixture) intent(auditID, role string, number int) (auditstore.Execution, error) {
	inputs := map[string]auditstore.ExactArtifact{"source": f.inputs["source"]}
	if role == "refine-api" {
		inputs["existing_openapi"] = f.accepted[auditID]["api"].Retained
	}
	executionID := auditdomain.DeterministicID("prepare", auditID, role, strconv.Itoa(number))
	manifestRevision := "manifest-revision"
	execution, _, err := f.store.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{
		Claim: f.claims[auditID], ExecutionID: executionID, Role: auditstore.ExecutionPrepare, WorkflowRole: role,
		LaneClass: resolved.LaneClassModel, RoleAttempt: &number, SubmissionKey: executionID,
		Manifest: auditstore.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(auditID),
			Name: "empty-manifest", Revision: &manifestRevision}, Digest: auditHandlerDigest("manifest")},
		RequestDigest: auditHandlerDigest(executionID),
		Preparation:   &auditstore.PreparationSnapshot{Inputs: inputs, Parameters: map[string]string{}},
	})
	return execution, err
}

func (f *preparedListFixture) current(auditID string) auditstore.Audit {
	f.t.Helper()
	audit, err := f.store.Get(f.ctx, f.owner, auditID)
	if err != nil {
		f.t.Fatal(err)
	}
	return audit
}

// TestSharedPostgresAuditListsHydratePreparationInFixedStatements serves the
// owner-wide and Project Audit lists over a full page of preparing Audits and
// a page that mixes every phase, compares each response with the per-row
// hydration the lists used before, and changes and purges listed Audits
// while a page is read (V407-004, V371-009).
func TestSharedPostgresAuditListsHydratePreparationInFixedStatements(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	trace := &auditListTrace{}
	tracedConfig := pool.Config()
	tracedConfig.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, tracedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	const owner, mixedProject, fullProject = "user-1", "project-mixed", "project-full"
	f := newPreparedListFixture(t, ctx, pool, owner, mixedProject, fullProject)

	// The mixed Project holds every phase and every preparation status.
	preparing := []string{"mixed-pending", "mixed-intent", "mixed-submitted", "mixed-retrying", "mixed-exhausted",
		"mixed-accepted", "mixed-inventory", "mixed-paused", "mixed-failed"}
	for _, auditID := range preparing {
		f.start(mixedProject, auditID)
	}
	f.claim(len(preparing))
	f.attempt(mixedProject, "mixed-intent", "generate-api", 1, "intent")
	f.attempt(mixedProject, "mixed-submitted", "generate-api", 1, "submitted")
	f.attempt(mixedProject, "mixed-retrying", "generate-api", 1, "failed")
	f.attempt(mixedProject, "mixed-exhausted", "generate-api", 1, "failed")
	f.attempt(mixedProject, "mixed-exhausted", "generate-api", 2, "failed")
	f.attempt(mixedProject, "mixed-accepted", "generate-api", 1, "succeeded")
	f.attempt(mixedProject, "mixed-accepted", "refine-api", 1, "intent")
	f.attempt(mixedProject, "mixed-inventory", "generate-api", 1, "succeeded")
	f.attempt(mixedProject, "mixed-inventory", "refine-api", 1, "succeeded")
	if _, err := f.store.CompletePreparation(ctx, f.claims["mixed-inventory"], f.current("mixed-inventory").Revision); err != nil {
		t.Fatal(err)
	}
	f.attempt(mixedProject, "mixed-paused", "generate-api", 1, "succeeded")
	if _, _, err := f.store.Transition(ctx, auditstore.TransitionParams{OwnerID: owner, AuditID: "mixed-paused",
		ExpectedRevision: f.current("mixed-paused").Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
		IdempotencyKey: "pause-mixed", RequestDigest: auditHandlerDigest("pause-mixed")}); err != nil {
		t.Fatal(err)
	}
	f.attempt(mixedProject, "mixed-failed", "generate-api", 1, "failed")
	if _, err := pool.Exec(ctx, `UPDATE audits SET state = 'failed', dispatch_state = 'closed', hold_state = 'released',
       finished_at = clock_timestamp(), stop_reason_code = 'preparation_failed', stop_reason_message = 'Preparation failed'
 WHERE audit_id = 'mixed-failed'`); err != nil {
		t.Fatal(err)
	}
	f.create(mixedProject, "mixed-draft")
	rounds := f.create(mixedProject, "mixed-rounds")
	taskRevision, sourceRevision := "task-r1", "source-r1"
	if _, _, err := f.store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: owner, AuditID: rounds.AuditID, ExpectedRevision: rounds.Revision, RoundID: "mixed-round", RoundOrdinal: 1,
		Manifest: auditstore.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "audits", Name: "mixed-worklist", Revision: &taskRevision},
			Digest: auditHandlerDigest("worklist")},
		BaselineSnapshot: f.baseline, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "rounds", RequestDigest: auditHandlerDigest("rounds"),
		Items: []auditstore.MaterializedItem{{ItemID: "mixed-item", ItemKey: "mixed-item", Kind: "checklist", SubjectKey: "subject",
			Task: auditstore.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "audits", Name: "mixed-task", Revision: &taskRevision},
				Digest: auditHandlerDigest("task")},
			Origin: auditstore.ItemOrigin{Schema: auditstore.ItemOriginSchema,
				SourceRef:           &contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &sourceRevision},
				SourceContentDigest: auditHandlerDigest("source"), SourceMediaType: "application/json",
				CanonicalInventoryDigest: auditHandlerDigest("inventory"), EntryKey: "mixed-item", EntryVersion: "1"},
			WorkflowRole: "scan", InitialState: auditstore.ItemReady,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}}},
	}); err != nil {
		t.Fatal(err)
	}
	wantPhases := map[string]auditdomain.AuditPhase{"mixed-draft": auditdomain.AuditPhaseNotStarted, "mixed-rounds": auditdomain.AuditPhaseRounds,
		"mixed-inventory": auditdomain.AuditPhaseInventory, "mixed-paused": auditdomain.AuditPhasePreparing, "mixed-failed": auditdomain.AuditPhasePreparing}
	for auditID, phase := range wantPhases {
		if current := f.current(auditID); current.Phase != phase {
			t.Fatalf("%s phase = %s, want %s", auditID, current.Phase, phase)
		}
	}
	// The newest 201 Audits are a full page of preparing Audits and one more.
	for index := range 201 {
		f.start(fullProject, fmt.Sprintf("full-%03d", index))
	}

	credentials := newFakeManagedCredentials()
	service, err := auditservice.New(auditservice.Options{Pool: traced, Profiles: f.catalog, CredentialGuard: credentials,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	h := auditTestHandler(service)
	h.Projects = projectstore.NewPostgresStore(traced)
	type listPage struct {
		Items []json.RawMessage `json:"items"`
		Page  pageInfoResponse  `json:"page"`
	}
	list := func(t *testing.T, projectID, query string) (*httptest.ResponseRecorder, listPage, int) {
		t.Helper()
		path := "/v1/audits"
		if projectID != "" {
			path = "/v1/projects/" + projectID + "/audits"
		}
		request := auditAuthenticatedRequest(http.MethodGet, path+"?"+query, nil)
		request.SetPathValue("projectId", projectID)
		trace.take()
		response := httptest.NewRecorder()
		h.listProjectAudits(response, request)
		statements := trace.take()
		var page listPage
		if response.Code == http.StatusOK {
			if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
				t.Fatal(err)
			}
		}
		return response, page, statements
	}
	// legacy builds the items of a page the way the lists did before
	// V407-004: per row, preparing and inventory items through the detail
	// read model. It returns them encoded and the statements they took.
	legacy := func(t *testing.T, page listPage) ([]byte, int) {
		t.Helper()
		request := auditAuthenticatedRequest(http.MethodGet, "/v1/audits", nil)
		listed := make([]auditstore.Audit, 0, len(page.Items))
		for _, item := range page.Items {
			var identity struct {
				AuditID string `json:"auditId"`
			}
			if err := json.Unmarshal(item, &identity); err != nil {
				t.Fatal(err)
			}
			listed = append(listed, f.current(identity.AuditID))
		}
		trace.take()
		items := make([]auditResponse, 0, len(listed))
		for _, audit := range listed {
			item, err := auditReadModel(audit)
			if err == nil && audit.Phase.PrecedesRounds() {
				item, err = h.readAuditDetail(request, audit)
			}
			if err != nil {
				t.Fatal(err)
			}
			items = append(items, item)
		}
		statements := trace.take()
		encoded, err := json.Marshal(items)
		if err != nil {
			t.Fatal(err)
		}
		return encoded, statements
	}
	document := loadPublicOpenAPI(t)
	// conforms validates a list response against the public AuditPage schema.
	conforms := func(t *testing.T, response *httptest.ResponseRecorder) {
		t.Helper()
		var wire any
		if err := json.Unmarshal(response.Body.Bytes(), &wire); err != nil {
			t.Fatal(err)
		}
		if err := document.Components.Schemas["AuditPage"].Value.VisitJSON(wire, openapi3.EnableJSONSchema2020()); err != nil {
			t.Fatalf("list response does not conform to AuditPage: %v", err)
		}
	}
	encodedItems := func(t *testing.T, page listPage) []byte {
		t.Helper()
		encoded, err := json.Marshal(page.Items)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}

	var nextCursor string
	t.Run("full page of preparing Audits", func(t *testing.T) {
		response, page, statements := list(t, "", "limit=200")
		if response.Code != http.StatusOK || len(page.Items) != 200 || !page.Page.HasMore || page.Page.NextCursor == nil {
			t.Fatalf("full page = %d %d items %+v", response.Code, len(page.Items), page.Page)
		}
		nextCursor = *page.Page.NextCursor
		conforms(t, response)
		for _, item := range page.Items {
			if !bytes.Contains(item, []byte(`"phase":"preparing"`)) || !bytes.Contains(item, []byte(`"preparation":{"roles":{"generate-api":`)) {
				t.Fatalf("full page item lost its preparation: %s", item)
			}
		}
		want, legacyStatements := legacy(t, page)
		if !bytes.Equal(encodedItems(t, page), want) {
			t.Fatalf("full page items differ from per-row hydration:\n got %s\nwant %s", encodedItems(t, page), want)
		}
		// BEGIN, the Audit list statement, one hydration statement and COMMIT,
		// whatever the number of preparing Audits; per-row hydration took
		// several statements per row.
		if statements != 4 || legacyStatements <= len(page.Items) {
			t.Fatalf("full page took %d statements; per-row hydration took %d", statements, legacyStatements)
		}
		t.Logf("200 preparing Audits: %d statements (per-row hydration: %d)", statements, legacyStatements)
		_, single, singleStatements := list(t, "", "limit=1")
		if len(single.Items) != 1 || singleStatements != statements {
			t.Fatalf("one preparing Audit took %d statements, a full page %d", singleStatements, statements)
		}
	})

	t.Run("page mixing every phase", func(t *testing.T) {
		response, page, statements := list(t, "", "limit=200&cursor="+url.QueryEscape(nextCursor))
		if response.Code != http.StatusOK || len(page.Items) != 12 || page.Page.HasMore || statements != 4 {
			t.Fatalf("mixed page = %d %d items %+v in %d statements", response.Code, len(page.Items), page.Page, statements)
		}
		conforms(t, response)
		want, _ := legacy(t, page)
		if !bytes.Equal(encodedItems(t, page), want) {
			t.Fatalf("mixed page items differ from per-row hydration:\n got %s\nwant %s", encodedItems(t, page), want)
		}
		projectResponse, projectPage, projectStatements := list(t, mixedProject, "limit=50")
		if projectResponse.Code != http.StatusOK || len(projectPage.Items) != 11 || projectStatements != 5 {
			t.Fatalf("Project page = %d %d items in %d statements", projectResponse.Code, len(projectPage.Items), projectStatements)
		}
		conforms(t, projectResponse)
		want, _ = legacy(t, projectPage)
		if !bytes.Equal(encodedItems(t, projectPage), want) {
			t.Fatalf("Project page items differ from per-row hydration:\n got %s\nwant %s", encodedItems(t, projectPage), want)
		}
		statuses := map[string]string{}
		for _, item := range projectPage.Items {
			var listed struct {
				AuditID            string                    `json:"auditId"`
				Phase              auditdomain.AuditPhase    `json:"phase"`
				Preparation        *auditPreparationResponse `json:"preparation"`
				GeneratedInventory json.RawMessage           `json:"generatedInventory"`
			}
			if err := json.Unmarshal(item, &listed); err != nil {
				t.Fatal(err)
			}
			hydrated := listed.Phase.PrecedesRounds()
			if (listed.Preparation != nil) != hydrated || listed.GeneratedInventory != nil {
				t.Fatalf("%s list item preparation=%t generatedInventory=%s", listed.AuditID, listed.Preparation != nil, listed.GeneratedInventory)
			}
			if !hydrated {
				continue
			}
			statuses[listed.AuditID] = string(listed.Preparation.Roles["generate-api"].Status) + "/" + string(listed.Preparation.Roles["refine-api"].Status)
			// Every item carries the preparation its detail read shows.
			detail := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/"+listed.AuditID, nil)
			detail.SetPathValue("auditId", listed.AuditID)
			detailed := httptest.NewRecorder()
			h.getAudit(detailed, detail)
			var read struct {
				Preparation json.RawMessage `json:"preparation"`
			}
			got, _ := json.Marshal(listed.Preparation)
			if detailed.Code != http.StatusOK || json.Unmarshal(detailed.Body.Bytes(), &read) != nil || !bytes.Equal(read.Preparation, got) {
				t.Fatalf("%s detail preparation = %d %s, list %s", listed.AuditID, detailed.Code, read.Preparation, got)
			}
		}
		wantStatuses := map[string]string{
			"mixed-pending": "pending/pending", "mixed-intent": "running/pending", "mixed-submitted": "running/pending",
			"mixed-retrying": "running/pending", "mixed-exhausted": "failed/pending", "mixed-accepted": "accepted/running",
			"mixed-inventory": "accepted/accepted", "mixed-paused": "accepted/pending", "mixed-failed": "failed/pending",
		}
		for auditID, want := range wantStatuses {
			if statuses[auditID] != want {
				t.Fatalf("preparation statuses = %v, want %v", statuses, wantStatuses)
			}
		}
		if _, drafts, draftStatements := list(t, "", "state=draft"); len(drafts.Items) != 1 || draftStatements != 3 {
			t.Fatalf("a page without preparing Audits took %d statements for %d items", draftStatements, len(drafts.Items))
		}
	})

	t.Run("selection runs after the snapshot commits", func(t *testing.T) {
		calls, callsAtCommit := 0, -1
		prepared := func(audit auditstore.Audit) bool {
			calls++
			return auditListHydratesPreparation(audit)
		}
		trace.mu.Lock()
		trace.marker, trace.interleaved = "commit", false
		trace.interleave = func(*pgx.Conn) error {
			callsAtCommit = calls
			return nil
		}
		trace.mu.Unlock()
		trace.take()
		audits, preparation, err := service.ListPage(ctx, auditstore.ListParams{OwnerID: owner, VisibleProjects: true,
			Limit: pagelimit.FetchRows(pagelimit.MaxRows)}, prepared)
		statements := trace.take()
		if err != nil || !trace.interleaved || trace.err != nil {
			t.Fatalf("full fetch = %v; commit seen=%t err=%v", err, trace.interleaved, trace.err)
		}
		// The selection, which decodes every pinned profile, runs only once
		// the transaction has committed, never between its statements.
		if callsAtCommit != 0 || calls != len(audits) || len(audits) != pagelimit.MaxFetchRows ||
			len(preparation) != len(audits) || statements != 4 {
			t.Fatalf("selection ran %d times before COMMIT and %d in all for %d Audits, %d preparations, %d statements",
				callsAtCommit, calls, len(audits), len(preparation), statements)
		}
		// The selection may pick only Audits whose phase precedes Rounds.
		drafts := auditstore.ListParams{OwnerID: owner, VisibleProjects: true, States: []auditstore.AuditState{auditstore.AuditDraft}, Limit: 2}
		if _, _, err := service.ListPage(ctx, drafts, func(auditstore.Audit) bool { return true }); err == nil {
			t.Fatal("a selection beyond the preparing and inventory phases was accepted")
		}
		if _, _, err := service.ListPage(ctx, drafts, nil); err == nil {
			t.Fatal("a page without a selection was accepted")
		}
	})

	t.Run("page read from one snapshot", func(t *testing.T) {
		before, _, _ := list(t, mixedProject, "limit=50")
		if before.Code != http.StatusOK {
			t.Fatalf("Project page = %d %s", before.Code, before.Body.String())
		}
		// Between the list statement and the hydration statement, the owner
		// pauses one listed preparing Audit, another one starts its next
		// prepare attempt and a third is cancelled, drained and purged.
		trace.mu.Lock()
		trace.marker, trace.interleaved = "WITH ORDINALITY AS listed", false
		trace.interleave = func(*pgx.Conn) error {
			if _, err := f.intent("mixed-retrying", "generate-api", 2); err != nil {
				return err
			}
			paused, err := f.store.Get(ctx, owner, "mixed-accepted")
			if err != nil {
				return err
			}
			if _, _, err := f.store.Transition(ctx, auditstore.TransitionParams{OwnerID: owner, AuditID: paused.AuditID,
				ExpectedRevision: paused.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
				IdempotencyKey: "pause-during-list", RequestDigest: auditHandlerDigest("pause-during-list")}); err != nil {
				return err
			}
			claim := f.claims["mixed-pending"]
			purged, err := f.store.Get(ctx, owner, claim.AuditID)
			if err != nil {
				return err
			}
			if _, _, err := f.store.Transition(ctx, auditstore.TransitionParams{OwnerID: owner, AuditID: purged.AuditID,
				ExpectedRevision: purged.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditCancelling,
				IdempotencyKey: "cancel-during-list", RequestDigest: auditHandlerDigest("cancel-during-list")}); err != nil {
				return err
			}
			if _, released, err := f.store.ReleaseDispatchHold(ctx, claim); err != nil || !released {
				return fmt.Errorf("release purged Audit hold: %t %w", released, err)
			}
			if purged, err = f.store.Get(ctx, owner, claim.AuditID); err != nil {
				return err
			}
			if _, err := f.store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: purged.Revision,
				ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting}); err != nil {
				return err
			}
			for range 10 {
				if err := f.store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(claim.AuditID)); err != nil {
					return err
				}
				if _, err := f.store.Get(ctx, owner, claim.AuditID); errors.Is(err, auditstore.ErrNotFound) {
					return nil
				} else if err != nil {
					return err
				}
			}
			return errors.New("the purged Audit outlived its purge batches")
		}
		trace.mu.Unlock()
		during, _, statements := list(t, mixedProject, "limit=50")
		if !trace.interleaved || trace.err != nil {
			t.Fatalf("interleaving ran=%t err=%v", trace.interleaved, trace.err)
		}
		// The page and every preparation come from the list's snapshot: the
		// response is the one served before either change.
		if during.Code != http.StatusOK || !bytes.Equal(during.Body.Bytes(), before.Body.Bytes()) || statements != 5 {
			t.Fatalf("Project page across a pause and a purge = %d in %d statements:\n got %s\nwant %s",
				during.Code, statements, during.Body.String(), before.Body.String())
		}
		after, page, _ := list(t, mixedProject, "limit=50")
		if after.Code != http.StatusOK || len(page.Items) != 10 || bytes.Contains(after.Body.Bytes(), []byte(`"auditId":"mixed-pending"`)) {
			t.Fatalf("Project page after the purge = %d %d items", after.Code, len(page.Items))
		}
		conforms(t, after)
		want, _ := legacy(t, page)
		if !bytes.Equal(encodedItems(t, page), want) || !bytes.Contains(want, []byte(`"auditId":"mixed-accepted"`)) {
			t.Fatalf("Project page after the pause differs from per-row hydration:\n got %s\nwant %s", encodedItems(t, page), want)
		}
		if paused := f.current("mixed-accepted"); paused.State != auditstore.AuditPaused ||
			!bytes.Contains(after.Body.Bytes(), []byte(`"status":"running","attempts":2`)) {
			t.Fatalf("mixed-accepted state = %s; the next attempt is missing: %s", paused.State, after.Body.String())
		}
	})

	t.Run("storage lost before hydration", func(t *testing.T) {
		trace.mu.Lock()
		trace.marker, trace.interleaved = "WITH ORDINALITY AS listed", false
		trace.interleave = func(conn *pgx.Conn) error {
			pid := conn.PgConn().PID()
			if _, err := pool.Exec(ctx, `SELECT pg_terminate_backend($1)`, pid); err != nil {
				return err
			}
			for range 500 {
				var alive bool
				if err := pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid = $1)`, pid).Scan(&alive); err != nil || !alive {
					return err
				}
				time.Sleep(10 * time.Millisecond)
			}
			return fmt.Errorf("backend %d outlived its termination", pid)
		}
		trace.mu.Unlock()
		response, _, _ := list(t, "", "limit=200")
		if !trace.interleaved || trace.err != nil {
			t.Fatalf("interleaving ran=%t err=%v", trace.interleaved, trace.err)
		}
		var failure errorResponse
		if response.Code != http.StatusInternalServerError || json.Unmarshal(response.Body.Bytes(), &failure) != nil ||
			failure.Code != "internal_error" || strings.Contains(response.Body.String(), `"items"`) {
			t.Fatalf("list after losing storage = %d %s", response.Code, response.Body.String())
		}
	})
}
