package postgres_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestPostgresMaterialSetSourcePurgeRacePreservesForksOrRejectsPreparation(t *testing.T) {
	for _, purgeFirst := range []bool{false, true} {
		t.Run(fmt.Sprint("purge-first=", purgeFirst), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			storage := artifactpgtest.NewStorage()
			service := NewService(storage.Repository(pool))
			projectID := "material-project"
			if _, _, e := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Materials", IdempotencyKey: "materials", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); e != nil {
				t.Fatal(e)
			}
			runs := runstore.NewPostgresStore(pool, storage)
			if _, e := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "material-run", OwnerID: "user-1", ProjectID: &projectID, WorkflowName: "set-reader", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); e != nil {
				t.Fatal(e)
			}
			project, _ := service.Project(projectID)
			written, e := project.Write(ctx, ArtifactRef{Namespace: "reports", Name: "source"}, Payload{MediaType: "text/plain", Data: []byte("exact retained source")}, nil)
			if e != nil {
				t.Fatal(e)
			}
			metadata, e := project.Metadata(ctx, written.Ref)
			if e != nil {
				t.Fatal(e)
			}
			set := contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: []contracts.MaterialDescriptor{{ScopeKind: "project", ScopeID: projectID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}}}
			raw, _ := json.Marshal(set)
			selected, e := project.Write(ctx, ArtifactRef{Namespace: "selections", Name: "set"}, Payload{MediaType: "application/json", Data: raw}, nil)
			if e != nil {
				t.Fatal(e)
			}
			scope, _ := ProjectScope(projectID)
			params := MaterialSetInput{Authority: MaterialAuthority{OwnerID: "user-1", ProjectID: projectID}, Source: scope, Ref: selected.Ref, RunID: "material-run", Slot: "materials", MediaTypes: []string{"text/plain"}, Required: true, Budget: &contracts.MaterialBudget{}}
			purge, e := pool.Begin(ctx)
			if e != nil {
				t.Fatal(e)
			}
			defer purge.Rollback(context.Background())
			purgeSource := func() error {
				p, e := storage.Purger(purge)
				if e != nil {
					return e
				}
				for range 4 {
					done, e := p.PurgeProject(ctx, projectID)
					if e != nil {
						return e
					}
					if done {
						return purge.Commit(ctx)
					}
				}
				return fmt.Errorf("source purge did not settle")
			}
			if purgeFirst {
				if e := purgeSource(); e != nil {
					t.Fatal(e)
				}
				if _, e := service.MaterializeInputSet(ctx, params); !IsExactReferenceRejection(e) {
					t.Fatalf("deleted source accepted or not permanent: %v", e)
				}
				run, _ := service.Run("material-run")
				if _, e := run.Metadata(ctx, ArtifactRef{Namespace: "inputs", Name: "materials"}); e == nil {
					t.Fatal("partial input after source deletion")
				}
				return
			}
			fork, e := pool.Begin(ctx)
			if e != nil {
				t.Fatal(e)
			}
			defer fork.Rollback(context.Background())
			prepared, e := NewService(storage.Repository(fork)).MaterializeInputSet(ctx, params)
			if e != nil {
				t.Fatal(e)
			}
			done := make(chan error, 1)
			go func() { done <- purgeSource() }()
			waitArtifactLock(t, ctx, pool, purge.Conn().PgConn().PID())
			if e := fork.Commit(ctx); e != nil {
				t.Fatal(e)
			}
			if e := <-done; e != nil {
				t.Fatal(e)
			}
			run, _ := service.Run("material-run")
			view, e := run.Read(ctx, prepared.Ref)
			if e != nil {
				t.Fatal(e)
			}
			local, e := contracts.DecodeMaterialSet(view.Payload.Data)
			if e != nil || len(local.Materials) != 1 {
				t.Fatal(local, e)
			}
			read, e := run.Read(ctx, local.Materials[0].Ref)
			if e != nil || string(read.Payload.Data) != "exact retained source" {
				t.Fatal("purge broke retained member", e)
			}
		})
	}
}
