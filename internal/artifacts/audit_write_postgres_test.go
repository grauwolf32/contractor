package artifacts_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestPostgresAuditArtifactWriterIsTheOnlyFrozenPath covers REV-R146 and
// REV-R149 on both blob backends: Server-generated Audit artifacts are created
// frozen through one writer, a replay of the same content returns the same
// exact revision without another revision, other content is a permanent
// collision, concurrent writers of one content converge, and generic Project
// writes cannot create or advance an audit-* binding.
func TestPostgresAuditArtifactWriterIsTheOnlyFrozenPath(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			t.Setenv("CONTRACTOR_TEST_ARTIFACT_BACKEND", backend)
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedArtifactPool(t, ctx)
			ctx = testBlobContext(t, ctx, pool)
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "audit-writer", OwnerID: "user-1", Kind: projectstore.KindProject,
				Name: "Audit writer", IdempotencyKey: "audit-writer", RequestDigest: "sha256:" + strings.Repeat("b", 64),
			}); err != nil {
				t.Fatal(err)
			}
			service := NewService(NewPostgresRepository(pool))
			project, err := service.Project("audit-writer")
			if err != nil {
				t.Fatal(err)
			}
			target := ArtifactRef{Namespace: "audit-writer-test", Name: "task-package"}
			payload := Payload{MediaType: "application/zip", Data: []byte("task package")}

			first, err := service.WriteAuditArtifact(ctx, "audit-writer", target, payload)
			if err != nil {
				t.Fatal(err)
			}
			metadata, err := project.Metadata(ctx, target)
			if err != nil || !metadata.Frozen || !metadata.Ref.SameExact(first.Ref) {
				t.Fatalf("written Audit binding = (%+v, %v), want frozen %+v", metadata, err, first.Ref)
			}
			replayed, err := service.WriteAuditArtifact(ctx, "audit-writer", target, payload)
			if err != nil || !replayed.Ref.SameExact(first.Ref) || replayed.Size != first.Size ||
				replayed.MediaType != first.MediaType {
				t.Fatalf("Audit write replay = (%+v, %v), want %+v", replayed, err, first)
			}
			for _, other := range []Payload{
				{MediaType: "application/zip", Data: []byte("other package")},
				{MediaType: "application/json", Data: payload.Data},
			} {
				if _, err := service.WriteAuditArtifact(ctx, "audit-writer", target, other); !errors.Is(err, ErrAuditArtifactCollision) ||
					!errors.Is(err, ErrArtifactIntegrity) {
					t.Fatalf("Audit write collision = %v", err)
				}
			}

			concurrent := ArtifactRef{Namespace: "audit-writer-test", Name: "worklist"}
			const writers = 4
			results := make([]WriteResult, writers)
			failures := make([]error, writers)
			var wait sync.WaitGroup
			for index := range writers {
				wait.Add(1)
				go func() {
					defer wait.Done()
					results[index], failures[index] = service.WriteAuditArtifact(ctx, "audit-writer", concurrent,
						Payload{MediaType: "application/zip", Data: []byte("shared worklist")})
				}()
			}
			wait.Wait()
			for index := range writers {
				if failures[index] != nil || !results[index].Ref.SameExact(results[0].Ref) {
					t.Fatalf("concurrent Audit write %d = (%+v, %v), first %+v", index, results[index], failures[index], results[0])
				}
			}

			var revisions int
			if err := pool.QueryRow(ctx, `
SELECT count(*) FROM artifact_binding_revisions
 WHERE scope_kind = 'project' AND scope_id = 'audit-writer'`).Scan(&revisions); err != nil || revisions != 2 {
				t.Fatalf("Audit binding revisions = (%d, %v), want 2", revisions, err)
			}

			if _, err := project.Write(ctx, ArtifactRef{Namespace: "audit-writer-test", Name: "generic"}, payload, nil); !errors.Is(err, ErrReservedNamespace) {
				t.Fatalf("generic create in an Audit namespace = %v", err)
			}
			if _, err := project.Write(ctx, target, payload, first.Ref.Revision); !errors.Is(err, ErrReservedNamespace) {
				t.Fatalf("generic advance of an Audit binding = %v", err)
			}
			if _, err := project.Write(ctx, ArtifactRef{Namespace: "sources", Name: "owner-upload"}, payload, nil); err != nil {
				t.Fatalf("owner upload outside the reserved prefix = %v", err)
			}
		})
	}
}

// TestPostgresExactReadVerifiesTheDescriptor covers the one exact read: the
// stored bytes, media type and size must match a complete descriptor, a
// digest-only descriptor pins the bytes, and any other content is an
// integrity failure.
func TestPostgresExactReadVerifiesTheDescriptor(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	user, err := NewService(NewPostgresRepository(pool)).User("owner")
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("exact content")
	written, err := user.Write(ctx, ArtifactRef{Namespace: "docs", Name: "exact"}, Payload{MediaType: "text/plain", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	complete := contracts.ExactContent(written.Ref, "text/plain", data)
	if read, err := user.ReadExact(ctx, complete); err != nil || string(read.Payload.Data) != string(data) {
		t.Fatalf("complete exact read = (%q, %v)", read.Payload.Data, err)
	}
	digestOnly := contracts.ExactArtifact{Ref: written.Ref, Digest: complete.Digest}
	if _, err := user.ReadExact(ctx, digestOnly); err != nil {
		t.Fatalf("digest-only exact read = %v", err)
	}
	for name, expected := range map[string]contracts.ExactArtifact{
		"digest":     {Ref: written.Ref, Digest: contracts.ExactContent(written.Ref, "", []byte("other")).Digest, MediaType: "text/plain", SizeBytes: complete.SizeBytes},
		"media type": {Ref: written.Ref, Digest: complete.Digest, MediaType: "text/markdown", SizeBytes: complete.SizeBytes},
		"size":       {Ref: written.Ref, Digest: complete.Digest, MediaType: "text/plain", SizeBytes: complete.SizeBytes + 1},
	} {
		if _, err := user.ReadExact(ctx, expected); !errors.Is(err, ErrArtifactIntegrity) ||
			!errors.Is(err, contracts.ErrExactArtifactMismatch) {
			t.Fatalf("%s mismatch exact read = %v", name, err)
		}
	}
	versionless := complete
	versionless.Ref = ArtifactRef{Namespace: "docs", Name: "exact"}
	if _, err := user.ReadExact(ctx, versionless); !errors.Is(err, ErrArtifactIntegrity) {
		t.Fatalf("versionless exact read = %v", err)
	}
}
