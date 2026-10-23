package runservice

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runrepeat"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresUserRulepackForksAcrossProjectsAndRejectsInvalidSourcesAtomically(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	root := t.TempDir()
	for _, directory := range []string{"agent-templates", "workflows", "instructions", "model-policies", "llm-gateways", "execution-configs", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, relative := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "scan-catalog", relative))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.HasPrefix(relative, "workflows/") {
			text = strings.Replace(text, "inputs: {}", "inputs:\n    rulepack: {required: false, sourceScope: user, mediaTypes: [application/yaml, application/vnd.contractor.nuclei-rulepack+zip]}", 1)
			text = strings.Replace(text, "context: {artifacts: {}}", "context: {artifacts: {rulepack: {namespace: inputs, name: rulepack, required: false}}}", 1)
		} else if strings.HasPrefix(relative, "agent-templates/") {
			text = strings.Replace(text, "url: {source: parameter, name: target}", "url: {source: parameter, name: target}\n      rulepack_ref: {source: artifact, name: rulepack, optional: true}", 1)
		}
		if err := os.WriteFile(filepath.Join(root, relative), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	projects := projectstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	user, _ := artifactService.User("pack-owner")
	other, _ := artifactService.User("other-owner")
	pack := writeExact(t, ctx, user, "rules", "pack", "application/yaml", []byte("exact owner rules"))
	_ = writeExact(t, ctx, other, "rules", "pack", "application/yaml", []byte("wrong owner rules"))
	wrongOwner := writeExact(t, ctx, other, "rules", "other", "application/yaml", []byte("not accessible"))
	badMedia := writeExact(t, ctx, user, "rules", "media", "application/zip", []byte("bad media"))
	tooBig := writeExact(t, ctx, user, "rules", "large", "application/yaml", bytes.Repeat([]byte("x"), scan.MaxNucleiRulepackFileBytes+1))
	service, err := New(Options{Runs: runs, Workflows: manager, LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{}, Projects: projects, PublicTransaction: postgresPublicTransaction(pool)})
	if err != nil {
		t.Fatal(err)
	}
	var saved PublicCreateParams
	for index := range 3 {
		var projectID *string
		if index > 0 {
			id := fmt.Sprintf("pack-project-%d", index)
			_, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "pack-owner", Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: testDigest("1")})
			if err != nil {
				t.Fatal(err)
			}
			projectID = &id
			projectArtifacts, _ := artifactService.Project(id)
			_ = writeExact(t, ctx, projectArtifacts, "rules", "pack", "application/yaml", []byte("project decoy"))
		}
		id := fmt.Sprintf("pack-run-%d", index)
		params := PublicCreateParams{OwnerID: "pack-owner", ProjectID: projectID, Workflow: "nuclei-target@1", Parameters: map[string]string{"target": "https://fixture.invalid/"}, Inputs: map[string]contracts.ArtifactRef{"rulepack": pack.Ref}, IdempotencyKey: id, NewRunID: func() (string, error) { return id, nil }}
		params.RequestDigest, err = PublicRequestDigest(params)
		if err != nil {
			t.Fatal(err)
		}
		result, err := service.CreatePublic(ctx, params)
		if err != nil || !result.Created {
			t.Fatalf("create: %+v, %v", result, err)
		}
		scope, _ := artifactService.Run(id)
		fork, err := scope.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"})
		if err != nil || string(fork.Payload.Data) != "exact owner rules" {
			t.Fatalf("fork=%+v err=%v", fork, err)
		}
		retained, err := runrepeat.Load(ctx, artifactService, id)
		if err != nil {
			t.Fatal(err)
		}
		source := retained.Inputs["rulepack"]
		if source.SourceScope != artifacts.ScopeUser || *source.Artifact.Revision != *pack.Ref.Revision {
			t.Fatalf("lost source authority: %+v", source)
		}
		saved = params
	}
	// Source advancement cannot change an admitted fork or the original key replay.
	if _, err := user.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: "pack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("new owner rules")}, pack.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	replay, err := service.CreatePublic(ctx, saved)
	if err != nil || !replay.Replayed {
		t.Fatalf("replay: %+v, %v", replay, err)
	}
	scope, _ := artifactService.Run(replay.Run.RunID)
	fork, err := scope.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"})
	if err != nil || string(fork.Payload.Data) != "exact owner rules" {
		t.Fatalf("replay changed fork: %v", err)
	}
	// Purging the populated Run artifact space removes its forks, never its owner source.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		t.Fatal(err)
	}
	if err := purger.PurgeRun(ctx, replay.Run.RunID); err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := user.Read(ctx, pack.Ref); err != nil {
		t.Fatalf("purge removed owner source: %v", err)
	}
	// Retire the source binding through the same guarded cleanup boundary used
	// by ArtifactStore purge tests. Existing Run revisions retain the shared blob.
	retire, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer retire.Rollback(ctx)
	for _, query := range []string{
		`SELECT set_config('contractor.lifecycle_purge','project',true)`,
		`DELETE FROM artifact_lineage WHERE source_scope_kind='user' AND source_scope_id='pack-owner' AND source_namespace='rules' AND source_name='pack'`,
		`DELETE FROM artifact_bindings WHERE scope_kind='user' AND scope_id='pack-owner' AND namespace='rules' AND name='pack'`,
		`DELETE FROM artifact_binding_revisions WHERE scope_kind='user' AND scope_id='pack-owner' AND namespace='rules' AND name='pack'`,
	} {
		if _, err := retire.Exec(ctx, query); err != nil {
			t.Fatal(err)
		}
	}
	if err := retire.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	for index := range 2 {
		remaining, _ := artifactService.Run(fmt.Sprintf("pack-run-%d", index))
		read, err := remaining.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"})
		if err != nil || string(read.Payload.Data) != "exact owner rules" {
			t.Fatalf("source deletion damaged fork: %v", err)
		}
	}
	if _, err := user.Read(ctx, pack.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("source retirement ineffective: %v", err)
	}
	if result, err := service.CreatePublic(ctx, saved); err != nil || !result.Replayed {
		t.Fatalf("source deletion broke keyed replay: %+v %v", result, err)
	}
	for index, ref := range []contracts.ArtifactRef{wrongOwner.Ref, badMedia.Ref, tooBig.Ref} {
		id := fmt.Sprintf("bad-pack-%d", index)
		params := PublicCreateParams{OwnerID: "pack-owner", Workflow: "nuclei-target@1", Parameters: map[string]string{"target": "https://fixture.invalid/"}, Inputs: map[string]contracts.ArtifactRef{"rulepack": ref}, IdempotencyKey: id, NewRunID: func() (string, error) { return id, nil }}
		params.RequestDigest, err = PublicRequestDigest(params)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := service.CreatePublic(ctx, params); err == nil {
			t.Fatal("invalid source accepted")
		}
		if _, err := runs.GetRun(ctx, id); !errors.Is(err, runstore.ErrNotFound) {
			t.Fatalf("invalid source committed Run: %v", err)
		}
		failedScope, _ := artifactService.Run(id)
		if _, err := failedScope.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatalf("invalid source left fork: %v", err)
		}
	}
}
