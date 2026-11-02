package public

import (
	"bytes"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/runservice"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestPostgresRunRepeatUsesExactCreationLineageWithoutMutation(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	fixture := newHandlerFixture(t)
	repository := artifactpostgres.NewRepository(pool)
	service := artifacts.NewService(repository)
	runs := runstore.NewPostgresStore(pool)
	projects := projectstore.NewPostgresStore(pool)
	h := &runRepeatHandlers{RunRepeatDependencies: RunRepeatDependencies{RepeatDrafts: runservice.NewRepeatReader(runservice.RepeatOptions{
		Runs: runs, Artifacts: service, Projects: projects,
		Config: fixture.configs, Credentials: fixture.credentials, RuntimeConfigs: fixture.runtimeConfigs,
	})}}
	// Include payload bytes, revisions, clocks, pins, lineage and Run events so
	// repeated projections cannot quietly repair history or create another Run.
	storedState := func() map[string]string {
		t.Helper()
		result := make(map[string]string)
		for _, table := range []string{
			"workflow_runs", "workflow_run_events", "projects",
			"artifact_scopes", "artifact_blobs", "artifact_versions", "artifact_binding_revisions",
			"artifact_bindings", "artifact_lineage",
		} {
			var state string
			if err := pool.QueryRow(ctx, `SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY to_jsonb(row)::text), '[]'::jsonb)::text FROM `+pgx.Identifier{table}.Sanitize()+` AS row`).Scan(&state); err != nil {
				t.Fatal(err)
			}
			result[table] = state
		}
		return result
	}
	for _, scope := range []string{"user", "project"} {
		t.Run(scope, func(t *testing.T) {
			var projectID *string
			sources, err := service.User("user-1")
			if scope == "project" {
				project, _, createErr := projects.Create(ctx, projectstore.CreateParams{
					ProjectID: "repeat-project", OwnerID: "user-1", Kind: projectstore.KindProject,
					Name: "Repeat Project", IdempotencyKey: "repeat-project", RequestDigest: "sha256:" + strings.Repeat("a", 64),
				})
				if createErr != nil {
					t.Fatal(createErr)
				}
				projectID = &project.ProjectID
				sources, err = service.Project(*projectID)
			}
			if err != nil {
				t.Fatal(err)
			}
			source, err := sources.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "original"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := sources.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "original"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("new current source")}, source.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			for _, variant := range []string{"current", "missing-required", "missing-edge", "ambiguous", "foreign-scope", "orphan-optional"} {
				t.Run(variant, func(t *testing.T) {
					runID := "repeat-" + scope + "-" + variant
					if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
						RequiredPlacementLabels: []string{"cloud/aws"},
						MetadataLabels:          map[string]string{"cloud": "aws"},
						LaneClass:               resolved.LaneClassModel,
						RunID:                   runID, OwnerID: "user-1", ProjectID: projectID, WorkflowName: "artifact-copy", WorkflowVersion: "1",
						WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: repeatTestSnapshot(t, fixture),
						Parameters: map[string]string{"objective": "original objective"}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
					}); err != nil {
						t.Fatal(err)
					}
					var fork artifacts.ForkResult
					if variant != "missing-required" {
						if variant == "foreign-scope" {
							foreign, _ := service.User("user-2")
							other, writeErr := foreign.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "foreign-" + scope}, artifacts.Payload{MediaType: "text/plain", Data: []byte("foreign")}, nil)
							if writeErr != nil {
								t.Fatal(writeErr)
							}
							fork, err = artifactstest.ForkInput(ctx, service, "user-2", other.Ref, runID, "source")
						} else if projectID == nil {
							fork, err = artifactstest.ForkInput(ctx, service, "user-1", source.Ref, runID, "source")
						} else {
							fork, err = artifactstest.ForkProjectInput(ctx, service, *projectID, source.Ref, runID, "source")
						}
						if err != nil {
							t.Fatal(err)
						}
					}
					working, _ := service.Run(runID)
					if variant == "current" || variant == "ambiguous" {
						advanced, writeErr := working.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("edited working copy")}, fork.TargetRef.Revision)
						if writeErr != nil {
							t.Fatal(writeErr)
						}
						if variant == "ambiguous" {
							scopeID := "user-1"
							if projectID != nil {
								scopeID = *projectID
							}
							_, err = pool.Exec(ctx, `INSERT INTO artifact_lineage (
 target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,
 source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind)
 VALUES ('run',$1,'inputs','source',$2,$3,$4,'sources','original',$5,'input_fork')`, runID, *advanced.Ref.Revision, scope, scopeID, *source.Ref.Revision)
							if err != nil {
								t.Fatal(err)
							}
						}
					}
					if variant == "missing-edge" {
						tx, txErr := pool.Begin(ctx)
						if txErr != nil {
							t.Fatal(txErr)
						}
						if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge','run',true)`); err != nil {
							t.Fatal(err)
						}
						if _, err := tx.Exec(ctx, `DELETE FROM artifact_lineage WHERE target_scope_kind='run' AND target_scope_id=$1`, runID); err != nil {
							t.Fatal(err)
						}
						if err := tx.Commit(ctx); err != nil {
							t.Fatal(err)
						}
					}
					if variant == "orphan-optional" {
						runScope, _ := artifacts.RunScope(runID)
						if _, err := service.Create(ctx, runScope, contracts.ArtifactRef{Namespace: "inputs", Name: "optional"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("unproven")}); err != nil {
							t.Fatal(err)
						}
					}

					if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture-complete"}); err != nil {
						t.Fatal(err)
					}
					before := storedState()
					var first []byte
					for range 2 {
						request := httptest.NewRequest(http.MethodGet, "/v1/runs/"+runID+"/repeat-draft", nil).WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: "user-1"}))
						request.SetPathValue("runId", runID)
						response := httptest.NewRecorder()
						serve(h.getRunRepeatDraft)(response, request)
						var repeat runRepeatDraftResponse
						decodeQueryResponse(t, response, &repeat)
						if response.Code != http.StatusOK || repeat.Authority != repeatAuthorityOrdinary {
							t.Fatalf("Repeat = %d %s", response.Code, response.Body.String())
						}
						if variant == "current" {
							if repeat.Draft == nil || len(repeat.Draft.RequiredPlacementLabels) != 1 || repeat.Draft.RequiredPlacementLabels[0] != "cloud/aws" {
								t.Fatalf("repeat lost pinned placement selection: %+v", repeat.Draft)
							}
							if repeat.Draft == nil || repeat.Draft.Parameters["objective"] != "original objective" || !sameOptionalString(repeat.ProjectID, projectID) {
								t.Fatalf("current request lost its draft: %+v", repeat)
							}
							input := repeat.Draft.Inputs["source"]
							if input.Artifact == nil || !input.Artifact.Equal(source.Ref) || input.Metadata == nil || input.Metadata.Current || input.SourceScope != artifacts.ScopeKind(scope) {
								t.Fatalf("current request lost exact source: %+v", input)
							}
						} else {
							code := "input_provenance_invalid"
							if repeat.Draft != nil || len(repeat.Notices) != 1 || repeat.Notices[0].Code != code || repeat.Notices[0].Severity != "blocking" {
								t.Fatalf("untrusted request produced a draft: %+v", repeat)
							}
						}
						if first != nil && !bytes.Equal(first, response.Body.Bytes()) {
							t.Fatal("repeated projection changed its response")
						}
						first = append([]byte(nil), response.Body.Bytes()...)
						if !reflect.DeepEqual(before, storedState()) {
							t.Fatal("Repeat modified retained state")
						}
					}
				})
			}
		})
	}
}

func TestPostgresRepeatRefusesChangedInputSourceScope(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	fixture := newHandlerFixture(t)
	service := artifacts.NewService(artifactpostgres.NewRepository(pool))
	runs := runstore.NewPostgresStore(pool)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "scope-drift-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Scope drift", IdempotencyKey: "scope-drift-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	user, _ := service.User("user-1")
	source, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("owner rules")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "scope-drift-run"
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: "user-1", ProjectID: &project.ProjectID, WorkflowName: "artifact-copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: repeatRulepackSnapshot(t, fixture), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.ForkInput(ctx, service, "user-1", source.Ref, runID, "rulepack"); err != nil {
		t.Fatal(err)
	}

	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture-complete"}); err != nil {
		t.Fatal(err)
	}
	h := &runRepeatHandlers{RunRepeatDependencies: RunRepeatDependencies{RepeatDrafts: runservice.NewRepeatReader(runservice.RepeatOptions{Runs: runs, Projects: projects, Artifacts: service, Config: fixture.configs, Credentials: fixture.credentials, RuntimeConfigs: fixture.runtimeConfigs})}}
	request := httptest.NewRequest(http.MethodGet, "/v1/runs/"+runID+"/repeat-draft", nil).WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: "user-1"}))
	request.SetPathValue("runId", runID)
	response := httptest.NewRecorder()
	serve(h.getRunRepeatDraft)(response, request)
	var repeat runRepeatDraftResponse
	decodeQueryResponse(t, response, &repeat)
	if response.Code != http.StatusOK || repeat.Draft == nil || repeat.Draft.Inputs["rulepack"].Status != repeatStatusUnavailable || !hasRepeatNotice(repeat.Notices, "input_source_scope_changed") {
		t.Fatalf("scope drift silently selected another source: %s", response.Body.String())
	}
}

func repeatTestSnapshot(t *testing.T, fixture handlerFixture) json.RawMessage {
	t.Helper()
	workflow, err := fixture.configs.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	optional := workflow.Inputs["source"]
	optional.Required = false
	workflow.Inputs["optional"] = optional
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func repeatRulepackSnapshot(t *testing.T, _ handlerFixture) json.RawMessage {
	t.Helper()
	root := t.TempDir()
	for _, directory := range []string{"agent-templates", "workflows", "instructions", "model-policies", "llm-gateways", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, relative := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("..", "..", "config", "testdata", "scan-catalog", relative))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.HasPrefix(relative, "workflows/") {
			text = strings.Replace(text, "inputs: {}", "inputs:\n    rulepack: {required: false, sourceScope: user, mediaTypes: [application/yaml]}", 1)
			text = strings.Replace(text, "context: {artifacts: {}}", "context: {artifacts: {rulepack: {namespace: inputs, name: rulepack, required: false}}}", 1)
		} else if strings.HasPrefix(relative, "agent-templates/") {
			text = strings.Replace(text, "url: {source: parameter, name: target}", "url: {source: parameter, name: target}\n      rulepack_ref: {source: artifact, name: rulepack, optional: true}", 1)
		}
		path := filepath.Join(root, relative)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := manager.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow.Ref = resolved.WorkflowRef{Name: "artifact-copy", Version: "1"}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
