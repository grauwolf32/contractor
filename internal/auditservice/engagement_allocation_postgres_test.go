package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// The baseline is created by ordinary Audit start, then consumed through the
// actual execution/Run/Stage ownership chain. Mutable inputs and Project
// policy cannot broaden the private task's pinned target assignment.
func TestEngagementStartToPrivateAllocationPins(t *testing.T) {
	for _, accessMode := range []string{"blackbox", "source-assisted"} {
		for _, controlled := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/controlled=%v", accessMode, controlled), func(t *testing.T) { testEngagementStartToPrivateAllocationPins(t, accessMode, controlled) })
		}
	}
}

func testEngagementStartToPrivateAllocationPins(t *testing.T, accessMode string, controlled bool) {
	f := newEngagementFixture(t, 1)
	// Dispatch admission compares its deadline to the database clock. No test
	// assertion depends on elapsed time; the authorized timestamp is compared
	// only to the exact baseline returned by start.
	f.service.now = time.Now
	document := f.document("http://127.0.0.1")
	if controlled {
		document.Policy.RequiredCapabilities = append(document.Policy.RequiredCapabilities, string(runtimesettings.RuntimeAdapterHTTPTarget))
	}
	if accessMode == "source-assisted" {
		requests := uint32(17)
		document.Budgets.TargetRequests = &requests
		document.AccessMode = accessMode
		source, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("print('fixture source')")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		metadata, err := f.store.Metadata(f.ctx, source.Ref)
		if err != nil {
			t.Fatal(err)
		}
		document.Inputs = append(document.Inputs, engagementcontract.Input{ID: "source", Role: "source", Artifact: source.Ref, ContentDigest: metadata.Digest})
		f.credential(t, "allocated-account", f.project.OwnerID)
		document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Target.Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "allocated-account", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	}
	draft, err := f.create(t, "allocation-engagement", document)
	if err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := auditprofile.DecodePinnedProgram(started.Audit.ProfileSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage())
	claim := claimAuditForPurge(t, f.ctx, store, draft.AuditID)
	snapshot, err := store.GetReconcileSnapshot(f.ctx, claim)
	if err != nil {
		t.Fatal(err)
	}
	accepted, err := f.service.AcceptInitialTasks(f.ctx, claim, snapshot)
	if err != nil || !accepted.Changed {
		t.Fatalf("initial work: %+v %v", accepted, err)
	}
	tasks, err := f.service.Tasks(f.ctx, auditstore.ListTasksParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, Limit: 2})
	if err != nil || len(tasks.Tasks) != 1 {
		t.Fatalf("tasks: %+v %v", tasks, err)
	}
	execution, _, err := store.SelectTasks(f.ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: []string{tasks.Tasks[0].Document.TaskID}})
	if err != nil {
		t.Fatal(err)
	}
	workflow := profile.Workflows["check"]
	workflowBytes, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "engagement-child-run"
	err = persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, createErr := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage()).CreateAuditRun(f.ctx, runstore.CreateAuditRunParams{CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: f.project.OwnerID, ProjectID: &f.project.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: "contractor/v1alpha1", WorkflowSnapshot: workflowBytes, Parameters: map[string]string{}, RuntimeConfig: baseline.RuntimeConfig, LaneClass: resolved.LaneClassModel}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey})
		if createErr != nil {
			return createErr
		}
		_, bindErr := auditstoretest.NewStore(tx, artifactpgtest.NewStorage()).BindRun(f.ctx, auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return bindErr
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	if _, err = runs.TransitionRun(f.ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err = runs.TransitionRun(f.ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stageBytes, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	stage, err := runs.CreateStageExecution(f.ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "engagement-stage", RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: "contractor/v1alpha1", StageSpecSnapshot: stageBytes, StageContextSchemaVersion: "contractor/v1alpha1", StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}}})
	if err != nil {
		t.Fatal(err)
	}
	protected := func(context.Context) ([]engagementcontract.ProtectedDestination, error) {
		if f.pool.Stat().AcquiredConns() != 0 {
			t.Fatal("infrastructure resolution held a database transaction")
		}
		return []engagementcontract.ProtectedDestination{{Hostname: "server.example", Addresses: []string{"10.22.1.2"}, Port: 443}}, nil
	}
	requirements, err := AllocationEngagementAdapterRequirements(f.ctx, f.pool, runID, stage.StageExecutionID, artifactpgtest.NewStorage())
	if err != nil || (len(requirements) == 1) != controlled {
		t.Fatalf("requirements=%v err=%v", requirements, err)
	}
	if controlled {
		if _, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID, nil, artifactpgtest.NewStorage()); err == nil {
			t.Fatal("controlled projection omitted protected destinations")
		}
	}
	first, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID, protected, artifactpgtest.NewStorage())
	if err != nil || first == nil {
		t.Fatalf("private projection=%+v %v", first, err)
	}
	pin := baseline.Engagement
	if controlled {
		if document.Budgets.TargetRequests == nil {
			if first.HTTP.Limits != nil {
				t.Fatal("omitted limits acquired a default")
			}
		} else if first.HTTP.Limits == nil || first.HTTP.Limits.TargetRequests == nil || *first.HTTP.Limits.TargetRequests != *document.Budgets.TargetRequests {
			t.Fatal("allocation lost the engagement's explicit HTTP limit")
		}
	}
	if first.EngagementDigest != pin.Artifact.Digest || first.PolicyDigest != pin.PolicyDigest || first.AuthorizedBy != f.project.OwnerID || !first.AuthorizedAt.Equal(pin.AuthorizedAt) || !first.Engagement.SameExact(pin.Artifact.Ref) || (first.HTTP != nil) != controlled {
		t.Fatalf("private projection drifted or exposed unassigned authority: %+v", first)
	}
	if controlled && (!reflect.DeepEqual(first.HTTP.Routes, document.Scope.Routes) || first.HTTP.Identities[0].ID != "anonymous" || first.HTTP.Identities[0].Credential != nil) {
		t.Fatal("controlled assignment drifted or exposed identity authority")
	}
	document.Target.Deployment = &engagementcontract.Deployment{Marker: "new mutable document"}
	replacement, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: pin.Artifact.Ref.Namespace, Name: pin.Artifact.Ref.Name}, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: replacement}, pin.Artifact.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	prohibited := contracts.ActiveChecksProhibited
	if _, err = projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: f.project.ProjectID, OwnerID: f.project.OwnerID, ExpectedRevision: f.project.Revision, Name: f.project.Name, ActiveChecks: &prohibited}); err != nil {
		t.Fatal(err)
	}
	second, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID, protected, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	firstBytes, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondBytes, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstBytes) != string(secondBytes) {
		t.Fatal("private metadata reread mutable Project or engagement")
	}
	*first.Engagement.Revision = "changed"
	first.Policy.RequiredCapabilities[0] = "changed"
	if controlled {
		first.HTTP.Routes[0].PathPrefixes[0] = "/changed"
		first.HTTP.Identities[0].Origins[0] = "http://changed.invalid"
		if first.HTTP.Limits != nil {
			*first.HTTP.Limits.TargetRequests = 99
		}
		if accessMode == "source-assisted" {
			first.HTTP.Identities[1].Credential.CredentialID = "changed"
		}
	}
	third, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID, protected, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	thirdBytes, err := json.Marshal(third)
	if err != nil || string(thirdBytes) != string(secondBytes) {
		t.Fatalf("mutating an allocation projection changed its retained authority: %v", err)
	}
	if _, err = ResolveAllocationEngagement(f.ctx, f.pool, "foreign-run", stage.StageExecutionID, nil, artifactpgtest.NewStorage()); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("foreign Run reached baseline: %v", err)
	}
	if _, err = ResolveAllocationEngagement(f.ctx, f.pool, runID, "foreign-stage", nil, artifactpgtest.NewStorage()); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("foreign Stage reached baseline: %v", err)
	}
}
