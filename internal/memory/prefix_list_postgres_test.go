package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestPostgresMemoryIgnoresOrdinaryBindingsAndDetectsQuotaOverflow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	createRunningMemoryStage(t, ctx, pool, "prefix-run", "prefix-stage")
	store, _ := NewPostgresStore(pool)
	namespace, _ := NewNamespace(store, Binding{RunID: "prefix-run", StageExecutionID: "prefix-stage", Namespace: "builder"})
	run, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run("prefix-run")
	_, err := run.Write(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: "ordinary"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `WITH bindings AS (
 INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision)
 SELECT 'run','prefix-run','builder','artifact_' || lpad(i::text,6,'0') || '_' || repeat('x',110),'bulk' FROM generate_series(1,7000)i
 RETURNING scope_kind,scope_id,namespace,name,current_revision
 ) INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
 SELECT scope_kind,scope_id,namespace,name,current_revision,(SELECT version_id FROM artifact_versions LIMIT 1) FROM bindings`)
	if err != nil {
		t.Fatal(err)
	}
	refs, err := store.List(ctx, namespace.binding)
	if err != nil || len(refs) != 0 {
		t.Fatalf("empty bounded view=%+v %v", refs, err)
	}
	first, err := namespace.WriteMemory(ctx, "note_0", "body", "", nil)
	if err != nil || first.Ordinal != 0 {
		t.Fatalf("first note=%+v %v", first, err)
	}
	for i := 1; i <= MaximumNotes; i++ {
		if i == MaximumNotes {
			refs, err = store.List(ctx, namespace.binding)
			if err != nil || len(refs) != MaximumNotes {
				t.Fatalf("at quota=%d %v", len(refs), err)
			}
			if _, err := namespace.WriteMemory(ctx, "overflow", "body", "", nil); memoryErrorCode(err) != CodeNamespaceFull {
				t.Fatalf("create at quota=%v", err)
			}
			if _, err := namespace.WriteMemory(ctx, "note_0", "replacement", "", nil); err != nil {
				t.Fatal(err)
			}
			previews, err := namespace.ListMemories(ctx)
			if err != nil || len(previews) != MaximumNotes {
				t.Fatalf("previews=%d %v", len(previews), err)
			}
		}
		name := fmt.Sprintf("note_%d", i)
		payload, err := Encode(StoredNote{SchemaVersion: SchemaVersion, Name: name, Content: "body", Ordinal: uint64(i)})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := run.Write(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: ArtifactNamePrefix + name}, artifacts.Payload{MediaType: MediaType, Data: payload}, nil); err != nil {
			t.Fatal(err)
		}
	}
	refs, err = store.List(ctx, namespace.binding)
	if err != nil || len(refs) != MaximumNotes+1 {
		t.Fatalf("overflow sentinel=%d %v", len(refs), err)
	}
	if _, err := namespace.ListMemories(ctx); memoryErrorCode(err) != CodeUnavailable {
		t.Fatalf("corrupt quota=%v", err)
	}
}
