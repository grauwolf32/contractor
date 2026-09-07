package artifacts_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
)

func TestPostgresPrefixListIsLiteralBoundedAndIsolated(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	service := NewService(NewPostgresRepository(pool))
	owner, _ := service.User("prefix-owner")
	foreign, _ := service.User("prefix-foreign")
	for _, item := range []struct {
		store           ScopedStore
		namespace, name string
	}{
		{owner, "builder", "memory.a"}, {owner, "builder", "memory.b"},
		{owner, "builder", "note_one"}, {owner, "builder", "noteXone"},
		{owner, "other", "memory.foreign"}, {foreign, "builder", "memory.foreign"},
	} {
		if _, err := item.store.Write(ctx, ArtifactRef{Namespace: item.namespace, Name: item.name}, Payload{MediaType: "text/plain", Data: []byte("body")}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// Many ordinary current bindings can point to one already valid immutable
	// version. Insert both relations in the same statement for the deferred FK.
	_, err := pool.Exec(ctx, `WITH bindings AS (
 INSERT INTO artifact_bindings (scope_kind,scope_id,namespace,name,current_revision)
 SELECT 'user','prefix-owner','builder','artifact_' || lpad(i::text,6,'0') || '_' || repeat('x',110),'bulk-rev' FROM generate_series(1,7000) i
 RETURNING scope_kind,scope_id,namespace,name,current_revision
 ) INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
 SELECT scope_kind,scope_id,namespace,name,current_revision,(SELECT version_id FROM artifact_versions LIMIT 1) FROM bindings`)
	if err != nil {
		t.Fatal(err)
	}
	refs, err := owner.ListPrefix(ctx, "builder", "memory.", 1)
	if err != nil || len(refs) != 1 || refs[0].Name != "memory.a" || refs[0].Revision != nil {
		t.Fatalf("bounded prefix = %+v, %v", refs, err)
	}
	refs, err = owner.ListPrefix(ctx, "builder", "memory.", 129)
	if err != nil || len(refs) != 2 {
		t.Fatalf("isolated prefix = %+v, %v", refs, err)
	}
	refs, err = owner.ListPrefix(ctx, "builder", "note_", 10)
	if err != nil || len(refs) != 1 || refs[0].Name != "note_one" {
		t.Fatalf("literal underscore = %+v, %v", refs, err)
	}
	namespace := "builder"
	refs, err = owner.List(ctx, &namespace)
	if err != nil || len(refs) != 7004 {
		t.Fatalf("legacy list count = %d, %v", len(refs), err)
	}
	for _, prefix := range []string{"", "memory%", "../memory", strings.Repeat("x", 129)} {
		if _, err := owner.ListPrefix(ctx, "builder", prefix, 129); err == nil {
			t.Fatalf("accepted prefix %q", prefix)
		}
	}
	for _, limit := range []int{-1, 0, 257} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			if _, err := owner.ListPrefix(ctx, "builder", "memory.", limit); err == nil {
				t.Fatal("accepted invalid limit")
			}
		})
	}
}
