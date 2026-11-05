package runservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestSharedPostgresInputMappingRejectsChangedOrdinaryRevision(t *testing.T) {
	for _, sameBytes := range []bool{false, true} {
		t.Run(fmt.Sprint("same-bytes=", sameBytes), func(t *testing.T) {
			f := newCommonTaskRunFixture(t, true)
			ctx := t.Context()
			made, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			service := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := service.Run(made.Run.RunID)
			before, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
			if err != nil {
				t.Fatal(err)
			}
			data := []byte("different input bytes")
			if sameBytes {
				data = before.Payload.Data
			}
			scope, _ := artifacts.RunScope(made.Run.RunID)
			changed, err := service.Replace(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: data}, *before.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			// A fresh owner has only the saved original fork and the new local revision.
			fresh := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			_, err = fresh.DescribeInputMaterials(ctx, made.Run.RunID, artifacts.MaterialAuthority{OwnerID: "owner", ProjectID: f.audit.ProjectID, AuditID: f.audit.AuditID}, []artifacts.MaterialInput{{Name: "source", Ref: changed.Ref, MediaTypes: []string{"text/plain"}}})
			if !errors.Is(err, blob.ErrIntegrity) {
				t.Fatalf("changed ordinary revision acquired original provenance: %v", err)
			}
		})
	}
}

func TestSharedPostgresInputMappingVerifiesDeclaredSetRewriteAfterRestart(t *testing.T) {
	for _, change := range []string{"none", "foreign-run", "lookalike-member", "missing-member"} {
		t.Run(change, func(t *testing.T) {
			f := newCommonTaskRunFixtureWithWorkflow(t, true, func(w *resolved.ResolvedWorkflow) {
				slot := w.Inputs["source"]
				slot.MaterialSet = true
				w.Inputs["source"] = slot
			})
			ctx := t.Context()
			made, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			service := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := service.Run(made.Run.RunID)
			value, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
			if err != nil {
				t.Fatal(err)
			}
			set, err := contracts.DecodeMaterialSet(value.Payload.Data)
			if err != nil {
				t.Fatal(err)
			}
			ref := value.Ref
			if change != "none" {
				switch change {
				case "foreign-run":
					set.Materials[0].ScopeID = "foreign-run"
				case "missing-member":
					set.Materials = []contracts.MaterialDescriptor{}
				case "lookalike-member":
					member, err := run.ReadExact(ctx, set.Materials[0].Exact())
					if err != nil {
						t.Fatal(err)
					}
					// The name and bytes match, but this is not a native input fork.
					fake, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: set.Materials[0].Ref.Name}, member.Payload, nil)
					if err != nil {
						t.Fatal(err)
					}
					set.Materials[0].Ref = fake.Ref
				}
				data, _ := json.Marshal(set)
				scope, _ := artifacts.RunScope(made.Run.RunID)
				written, err := service.Replace(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "application/json", Data: data}, *value.Ref.Revision)
				if err != nil {
					t.Fatal(err)
				}
				ref = written.Ref
			}
			fresh := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			mapping, err := fresh.DescribeInputMaterials(ctx, made.Run.RunID, artifacts.MaterialAuthority{OwnerID: "owner", ProjectID: f.audit.ProjectID, AuditID: f.audit.AuditID}, []artifacts.MaterialInput{{Name: "source", Ref: ref, MediaTypes: []string{"text/plain"}, Set: true, Required: true}})
			if change != "none" {
				if err == nil {
					t.Fatal("invalid set transformation acquired source provenance")
				}
				return
			}
			if err != nil || mapping["source"].Source.ScopeKind != "audit" || mapping["source"].Source.Digest != f.source.Digest || mapping["source"].Local.Digest != value.Exact().Digest {
				t.Fatalf("valid persisted set mapping was not restored: %+v %v", mapping, err)
			}
		})
	}
}
