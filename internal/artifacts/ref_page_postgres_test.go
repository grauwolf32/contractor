package artifacts_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	. "github.com/grauwolf32/contractor/internal/artifacts"
)

type sharedListCases struct {
	MaxLimit int `json:"maxLimit"`
	Bindings []struct {
		Namespace string `json:"namespace"`
		Name      string `json:"name"`
		Hidden    bool   `json:"hidden"`
	} `json:"bindings"`
}

func readSharedListCases(t *testing.T) sharedListCases {
	t.Helper()
	data, err := os.ReadFile("../../api/testdata/v1alpha1/artifact-list-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases sharedListCases
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.MaxLimit != MaxBindingListLimit || len(cases.Bindings) == 0 {
		t.Fatalf("shared list cases: maximum %d (Go %d), %d bindings", cases.MaxLimit, MaxBindingListLimit, len(cases.Bindings))
	}
	return cases
}

// seedRunBindings inserts current bindings that all point at one existing
// version; both relations are written in one statement for the deferred FK.
func seedRunBindings(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string, refs []ArtifactRef) {
	t.Helper()
	namespaces := make([]string, 0, len(refs))
	names := make([]string, 0, len(refs))
	for _, ref := range refs {
		namespaces = append(namespaces, ref.Namespace)
		names = append(names, ref.Name)
	}
	_, err := pool.Exec(ctx, `WITH bindings AS (
 INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
 SELECT 'run', $1, input.namespace, input.name, 'bulk-rev'
 FROM unnest($2::text[], $3::text[]) AS input(namespace, name)
 RETURNING scope_kind, scope_id, namespace, name, current_revision
) INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
 SELECT scope_kind, scope_id, namespace, name, current_revision, (SELECT version_id FROM artifact_versions LIMIT 1)
 FROM bindings`, runID, namespaces, names)
	if err != nil {
		t.Fatalf("seed bindings: %v", err)
	}
}

// pageAll follows every page of query and fails on a page over the limit, a
// short page before the end or a revision in a listed ref.
func pageAll(t *testing.T, ctx context.Context, store ScopedStore, query RefPageQuery) []string {
	t.Helper()
	var listed []string
	for pages := 0; ; pages++ {
		if pages > 10000 {
			t.Fatal("listing did not terminate")
		}
		page, err := store.ListRefs(ctx, query)
		if err != nil {
			t.Fatalf("list page %d: %v", pages, err)
		}
		if len(page.Refs) > query.Limit || page.More && len(page.Refs) != query.Limit {
			t.Fatalf("page %d holds %d refs for limit %d, more %t", pages, len(page.Refs), query.Limit, page.More)
		}
		for _, ref := range page.Refs {
			if ref.Revision != nil {
				t.Fatalf("listed ref has a revision: %+v", ref)
			}
			listed = append(listed, ref.Namespace+"/"+ref.Name)
		}
		if !page.More {
			return listed
		}
		last := page.Refs[len(page.Refs)-1]
		query.AfterNamespace, query.AfterName = last.Namespace, last.Name
	}
}

func TestPostgresRefPagesExcludeModelHiddenBindingsBeforeTheLimit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	createArtifactRun(t, ctx, pool, "run-page", true)
	createArtifactRun(t, ctx, pool, "run-foreign", true)
	service := NewService(NewPostgresRepository(pool))
	run, err := service.Run("run-page")
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := service.Run("run-foreign")
	if err != nil {
		t.Fatal(err)
	}
	seed := ArtifactRef{Namespace: "builder", Name: "seed"}
	for _, store := range []ScopedStore{run, foreign} {
		if _, err := store.Write(ctx, seed, Payload{MediaType: "text/plain", Data: []byte("seed")}, nil); err != nil {
			t.Fatal(err)
		}
	}
	cases := readSharedListCases(t)
	visible := map[string]bool{"builder/seed": true}
	all := map[string]bool{"builder/seed": true}
	var refs []ArtifactRef
	for _, binding := range cases.Bindings {
		refs = append(refs, ArtifactRef{Namespace: binding.Namespace, Name: binding.Name})
		all[binding.Namespace+"/"+binding.Name] = true
		if !binding.Hidden {
			visible[binding.Namespace+"/"+binding.Name] = true
		}
	}
	// Hidden names sort between visible ones, so an exclusion applied after
	// the LIMIT would leave short pages in the middle of the traversal.
	for i := 0; i < 600; i++ {
		refs = append(refs,
			ArtifactRef{Namespace: "builder", Name: fmt.Sprintf("http.body.%016x.%06d", i, i)},
			ArtifactRef{Namespace: "builder", Name: fmt.Sprintf("memory.note_%04d", i)},
		)
		all[fmt.Sprintf("builder/http.body.%016x.%06d", i, i)] = true
		all[fmt.Sprintf("builder/memory.note_%04d", i)] = true
	}
	for i := 0; i < 300; i++ {
		for _, ref := range []ArtifactRef{
			{Namespace: "builder", Name: fmt.Sprintf("a_%04d", i)},
			{Namespace: "builder", Name: fmt.Sprintf("z_%04d", i)},
		} {
			refs = append(refs, ref)
			all[ref.Namespace+"/"+ref.Name] = true
			visible[ref.Namespace+"/"+ref.Name] = true
		}
	}
	seedRunBindings(t, ctx, pool, "run-page", refs)
	seedRunBindings(t, ctx, pool, "run-foreign", []ArtifactRef{{Namespace: "builder", Name: "foreign-only"}})

	assertExactlyOnce := func(t *testing.T, listed []string, want map[string]bool) {
		t.Helper()
		seen := make(map[string]bool, len(listed))
		for _, key := range listed {
			if seen[key] || !want[key] {
				t.Fatalf("listed %q unexpectedly or more than once", key)
			}
			seen[key] = true
		}
		if len(seen) != len(want) {
			t.Fatalf("listed %d of %d bindings", len(seen), len(want))
		}
	}
	for _, limit := range []int{1, 7, MaxBindingListLimit} {
		t.Run(fmt.Sprintf("model visible limit %d", limit), func(t *testing.T) {
			assertExactlyOnce(t, pageAll(t, ctx, run, RefPageQuery{ModelVisible: true, Limit: limit}), visible)
		})
	}
	t.Run("all bindings", func(t *testing.T) {
		assertExactlyOnce(t, pageAll(t, ctx, run, RefPageQuery{Limit: 50}), all)
	})
	for _, namespace := range []string{"builder", "inputs", "outputs", "finding-proposals", "skills", "contractor-system", "absent"} {
		t.Run("namespace "+namespace, func(t *testing.T) {
			want := make(map[string]bool)
			for key := range visible {
				if prefix, _, _ := strings.Cut(key, "/"); prefix == namespace {
					want[key] = true
				}
			}
			value := namespace
			assertExactlyOnce(t, pageAll(t, ctx, run, RefPageQuery{Namespace: &value, ModelVisible: true, Limit: 9}), want)
		})
	}
	t.Run("foreign run", func(t *testing.T) {
		assertExactlyOnce(t, pageAll(t, ctx, foreign, RefPageQuery{ModelVisible: true, Limit: 5}),
			map[string]bool{"builder/seed": true, "builder/foreign-only": true})
	})
}
