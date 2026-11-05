package auditservice

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

type commonTaskServiceFixture struct {
	pool     *pgxpool.Pool
	service  *Service
	audit    auditstore.Audit
	metadata artifacts.Metadata
	pinned   auditprofile.PinnedProgram
}

func newCommonTaskServiceFixture(t *testing.T, configure func(*auditcontract.Program), workflowChanges ...func(*resolved.ResolvedWorkflow)) commonTaskServiceFixture {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(config *pgxpool.Config) { config.MaxConns = 1 }})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	if err := os.CopyFS(root, os.DirFS("../config/testdata/audit-common-catalog")); err != nil {
		t.Fatal(err)
	}
	if err := os.CopyFS(root, os.DirFS("../config/testdata/openapi-prepared-work")); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct{ Cases []struct{ Document string } }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "programs.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	program, err := auditcontract.DecodeProgram([]byte(fixture.Cases[0].Document))
	if err != nil {
		t.Fatal(err)
	}
	if configure != nil {
		configure(&program)
	}
	data, _ := json.Marshal(program)
	pinned, err := auditprofile.PinProgram(resolved.Selector{ID: "common", Version: "1"}, data, func(ref string) (resolved.ResolvedWorkflow, error) {
		workflow, err := catalog.Workflow(ref)
		if err == nil {
			for _, change := range workflowChanges {
				change(&workflow)
			}
		}
		return workflow, err
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "task-project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Task service", IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	project, _ := artifactService.Project("task-project")
	write, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected input")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := project.Metadata(ctx, write.Ref)
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := json.Marshal(pinned)
	limits := pinned.Program.Limits
	audit, _, err := ownerstore.New(pool, artifactpgtest.NewStorage()).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "common-audit", OwnerID: "owner", ProjectID: "task-project", Profile: auditstore.ProfileIdentity{Name: pinned.Ref.Name, Version: pinned.Ref.Version, Digest: pinned.Ref.Digest}, ProfileSnapshot: profile, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: limits.MaxIterations, BatchSize: limits.BatchSize, MaxItemsPerRound: limits.MaxTasksPerIteration, MaxItemsTotal: limits.MaxTasks, MaxSubmittedRuns: limits.MaxSubmittedRuns, MaxItemRunAttempts: limits.MaxAttempts, MaxEvidenceBytes: limits.MaxEvidenceBytes}, IdempotencyKey: "draft", RequestDigest: contentdigest.Bytes([]byte("draft"))})
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := EncodeBaseline(BaselineSnapshot{Schema: BaselineSchema, ActiveChecks: contracts.ActiveChecksProhibited, Labels: runlabels.RunMetadataLabels{}, RequiredPlacementLabels: []string{}, Inputs: map[string]auditstore.ExactArtifact{"source": metadata.Exact()}, RuntimeLabels: []string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), Skills: []agentskills.RunSkillSnapshot{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET state='paused',phase='rounds',started_at=clock_timestamp(),paused_at=clock_timestamp(),baseline_snapshot=$2,dispatch_state='closed' WHERE audit_id=$1`, audit.AuditID, baseline); err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, storage: artifactpgtest.NewStorage()}
	audit, err = service.Get(ctx, "owner", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	return commonTaskServiceFixture{pool: pool, service: service, audit: audit, metadata: metadata, pinned: pinned}
}

func TestSharedCommonTaskServiceAcceptsDefaultsAndExactOverrides(t *testing.T) {
	f := newCommonTaskServiceFixture(t, nil)
	ctx, service, audit, metadata := t.Context(), f.service, f.audit, f.metadata
	for _, explicit := range []bool{false, true} {
		current, err := service.Get(ctx, "owner", audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		key := "default"
		inputs := map[string]auditcontract.Material{}
		if explicit {
			key = "explicit"
			inputs["source"] = auditcontract.Material{ScopeKind: "project", ScopeID: audit.ProjectID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
		}
		params := TaskCreationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: key, RequestDigest: contentdigest.Bytes([]byte(key)), Work: []auditcontract.PreparedSelection{{Key: key, Workflow: "inspect", Objective: "Inspect exact selected input", Inputs: inputs}}}
		result, err := service.CreateTasks(ctx, params)
		if err != nil || result.Replayed || result.Acceptance.CreatedCount != 1 || len(result.Acceptance.TaskIDs) != 1 {
			t.Fatalf("create %s: %+v %v", key, result, err)
		}
		replayed, err := service.CreateTasks(ctx, params)
		if err != nil || !replayed.Replayed || replayed.Acceptance.TaskIDs[0] != result.Acceptance.TaskIDs[0] {
			t.Fatalf("replay %s: %+v %v", key, replayed, err)
		}
		task, err := service.Task(ctx, "owner", audit.AuditID, result.Acceptance.TaskIDs[0])
		if err != nil || task.Document.Inputs["source"].ScopeKind != "audit" || task.Document.Inputs["source"].Digest != metadata.Digest {
			t.Fatalf("retained input: %+v %v", task, err)
		}
	}
	page, err := service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 10})
	if err != nil || len(page.Tasks) != 2 || !page.Tasks[0].Document.Inputs["source"].Ref.SameExact(page.Tasks[1].Document.Inputs["source"].Ref) {
		t.Fatalf("retention did not converge: %+v %v", page, err)
	}
}
