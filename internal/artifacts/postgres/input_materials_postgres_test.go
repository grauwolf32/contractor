package postgres_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type selectedInputReads struct {
	artifacts.Repository
	payloads, metadata, batches int
	t                           *testing.T
}

func (r *selectedInputReads) Read(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.ReadResult, error) {
	r.payloads++
	return r.Repository.Read(ctx, scope, ref)
}
func (r *selectedInputReads) Metadata(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.Metadata, error) {
	r.metadata++
	return r.Repository.Metadata(ctx, scope, ref)
}
func (r *selectedInputReads) DescribeExact(ctx context.Context, scope artifacts.Scope, refs []artifacts.ArtifactRef) ([]contracts.ExactArtifact, error) {
	r.batches++
	if len(refs) > artifacts.MaxExactReadBatchSize {
		r.t.Fatal("unbounded metadata selection")
	}
	return r.Repository.DescribeExact(ctx, scope, refs)
}

func TestPostgresCheckedInputsReadOnceThenReturnExactPreparedDescriptors(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	base := artifacts.NewService(storage.Repository(pool))
	user, _ := base.User("user-1")
	createArtifactRun(t, ctx, pool, "prepared-materials", true)
	selected := []contracts.MaterialDescriptor{}
	for i := range 32 {
		raw := []byte(fmt.Sprint("bounded selected source ", i))
		written, err := user.Write(ctx, artifacts.ArtifactRef{Namespace: "reports", Name: fmt.Sprint("source-", i)}, artifacts.Payload{MediaType: "text/plain", Data: raw}, nil)
		if err != nil {
			t.Fatal(err)
		}
		exact := contracts.ExactContent(written.Ref, "text/plain", raw)
		selected = append(selected, contracts.MaterialDescriptor{ScopeKind: "user", ScopeID: "user-1", Ref: exact.Ref, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes})
	}
	raw, _ := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: selected})
	descriptor, err := user.Write(ctx, artifacts.ArtifactRef{Namespace: "selections", Name: "materials"}, artifacts.Payload{MediaType: "application/json", Data: raw}, nil)
	if err != nil {
		t.Fatal(err)
	}
	scope, _ := artifacts.UserScope("user-1")
	reads := &selectedInputReads{Repository: storage.Repository(pool), t: t}
	service := artifacts.NewService(reads)
	checked, err := service.CheckInputMaterials(ctx, artifacts.MaterialAuthority{OwnerID: "user-1"}, []artifacts.MaterialInput{{Name: "materials", Source: scope, Ref: descriptor.Ref, Set: true, Required: true, Research: true, MediaTypes: []string{"text/plain"}}, {Name: "singleton", Source: scope, Ref: selected[0].Ref, Research: true, MediaTypes: []string{"text/plain"}}}, &contracts.MaterialBudget{})
	if err != nil {
		t.Fatal(err)
	}
	if reads.payloads != 1 || reads.metadata != 0 || reads.batches != 2 {
		t.Fatalf("checking reread selected inputs: %+v", reads)
	}
	prepared, err := service.MaterializeInputs(ctx, "prepared-materials", checked)
	if err != nil {
		t.Fatal(err)
	}
	if reads.payloads != 1 || reads.metadata != 0 || reads.batches != 2 {
		t.Fatalf("materialization reconstructed checked inputs: %+v", reads)
	}
	if len(prepared.Materials) != 32 || len(prepared.Inputs) != 2 {
		t.Fatalf("wrong prepared cardinality: %+v", prepared)
	}
	run, _ := base.Run("prepared-materials")
	for _, exact := range prepared.Inputs {
		if _, err := run.ReadExact(ctx, exact); err != nil {
			t.Fatal(err)
		}
	}
	for key, m := range prepared.Materials {
		if contracts.MaterialHandle(m.Source) != key {
			t.Fatal("canonical source identity changed")
		}
		if _, err := run.ReadExact(ctx, m.Local.Exact()); err != nil {
			t.Fatal(err)
		}
	}
}
