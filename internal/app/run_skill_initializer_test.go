package app

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresRunSkillInitializerUsesCommittedExactSelection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAppPool(t, ctx)
	workflow := appSkillWorkflow(t)
	workflowSnapshot, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	owner, _ := service.User("owner-1")
	packageA := appSkillPackage(t, "Package A.")
	created, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: packageA},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		txService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		catalog, _ := agentskills.NewCatalog(txService)
		refs, err := config.WorkflowSkillRefs(workflow)
		if err != nil {
			return err
		}
		selected, err := catalog.SelectRunSources(ctx, "owner-1", refs)
		if err != nil {
			return err
		}
		if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
			RunID: "run-skill-init", OwnerID: "owner-1",
			WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowSnapshot,
			Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
		}); err != nil {
			return err
		}
		if err := store.SetRunSkillSelections(ctx, "run-skill-init", selected); err != nil {
			return err
		}
		return catalog.PinRunSources(ctx, "owner-1", "run-skill-init", selected)
	})
	if err != nil {
		t.Fatal(err)
	}
	packageB := appSkillPackage(t, "Package B.")
	if _, err := owner.Write(
		ctx,
		artifacts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: packageB},
		created.Ref.Revision,
	); err != nil {
		t.Fatal(err)
	}

	initialized, err := (&runSkillInitializer{pool: pool}).InitializeRunSkills(ctx, "run-skill-init")
	if err != nil || initialized.State != runstore.RunRunning ||
		len(initialized.SkillSnapshot) != 1 || initialized.SkillSnapshot[0].Artifact == nil ||
		initialized.SkillSnapshot[0].Source == nil ||
		*initialized.SkillSnapshot[0].Source.Revision != *created.Ref.Revision {
		t.Fatalf("initialized Run = (%+v, %v)", initialized, err)
	}
	run, _ := service.Run("run-skill-init")
	forked, err := run.Read(ctx, *initialized.SkillSnapshot[0].Artifact)
	if err != nil || string(forked.Payload.Data) != string(packageA) {
		t.Fatalf("pinned Run package = (%d bytes, %v)", len(forked.Payload.Data), err)
	}
	lineage, err := run.ListLineage(
		ctx, *initialized.SkillSnapshot[0].Artifact, artifacts.LineagePageQuery{Limit: 10},
	)
	if err != nil || len(lineage) != 1 || lineage[0].Source.Revision == nil ||
		*lineage[0].Source.Revision != *created.Ref.Revision {
		t.Fatalf("Run Skill lineage = (%+v, %v)", lineage, err)
	}
}

func appSkillWorkflow(t *testing.T) config.ResolvedWorkflow {
	t.Helper()
	root := copyAppConfigTree(t)
	path := filepath.Join(root, "agent-templates", "artifact_builder.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = bytes.Replace(
		data,
		[]byte("  sandboxProfile: local-workdir@1\n"),
		[]byte("  skills: [{namespace: skills, name: review}]\n  sandboxProfile: local-workdir@1\n"),
		1,
	)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	return workflow
}

func appSkillPackage(t *testing.T, description string) []byte {
	t.Helper()
	directory := filepath.Join(t.TempDir(), "review")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	document := "---\nname: review\ndescription: " + description + "\n---\n# Review\n"
	if err := os.WriteFile(filepath.Join(directory, "SKILL.md"), []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	payload, _, err := agentskills.PackageDirectory(directory)
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func copyAppConfigTree(t *testing.T) string {
	t.Helper()
	source := filepath.Clean("../config/testdata/valid")
	target := filepath.Join(t.TempDir(), "configs")
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		destination := filepath.Join(target, relative)
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(destination, data, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	return target
}

func isolatedAppPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}
