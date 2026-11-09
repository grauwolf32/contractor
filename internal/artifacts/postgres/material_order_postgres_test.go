package postgres_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestSharedMaterialKindMatchesUIFixtures(t *testing.T) {
	pool := isolatedArtifactPool(t, t.Context())
	raw := contractstest.ReadFile(t, "api", "testdata", "material-kind-order.json")
	var cases []struct {
		Namespace string
		MediaType string
		GitSource bool
		Rank      int
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 30 {
		t.Fatal("material classification fixture is incomplete")
	}
	for _, c := range cases {
		var rank int
		if err := pool.QueryRow(t.Context(), "SELECT contractor_material_rank($1,$2,$3)", c.Namespace, c.MediaType, c.GitSource).Scan(&rank); err != nil {
			t.Fatal(err)
		}
		if rank != c.Rank {
			t.Fatalf("%s %s git=%t: rank %d, want %d", c.Namespace, c.MediaType, c.GitSource, rank, c.Rank)
		}
	}
}

func TestPostgresProjectMaterialOrderPagesBeforeHydration(t *testing.T) {
	ctx := t.Context()
	pool := isolatedArtifactPool(t, ctx)
	const id = "priority-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: id, OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Priority", IdempotencyKey: id, RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	storage := artifactpgtest.NewStorage()
	service := NewService(storage.Repository(pool))
	project, _ := service.Project(id)
	write := func(namespace, name, media string) WriteResult {
		t.Helper()
		result, err := project.Write(ctx, ArtifactRef{Namespace: namespace, Name: name}, Payload{MediaType: media, Data: []byte(name)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	// The ordinary lexicographic first page consists entirely of generated
	// rows. Every useful namespace sorts after it, including a typed output.
	for i := 0; i < 1200; i++ {
		write("a-generated", fmt.Sprintf("evidence-%03d", i), "application/json")
	}
	write("sources", "code", "application/zip")
	write("z-output", "api", "application/vnd.oai.openapi+json")
	write("engagement", "scope", "application/vnd.contractor.engagement+json")
	write("likec4", "model", "text/vnd.likec4")
	write("docs", "guide", "text/markdown")
	write("diffs", "patch", "text/x-diff")
	// Git provenance is attached to a newly created version in its import
	// transaction. No fork can observe the unclassified intermediate state.
	scope, _ := ProjectScope(id)
	var imported WriteResult
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repo := storage.TxRepository(tx)
		target, _ := NewService(repo).Project(id)
		var err error
		imported, err = target.Write(ctx, ArtifactRef{Namespace: "z-imports", Name: "git"}, Payload{MediaType: "application/zip", Data: []byte("git")}, nil)
		if err != nil {
			return err
		}
		return repo.RecordGitSource(ctx, scope, imported.Ref, GitSource{RepositoryURL: "https://example.test/repo.git", ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC()})
	}); err != nil {
		t.Fatal(err)
	}

	// High-priority archived API specs must not crowd current material pages.
	archive := write("historical-results", "typed-archive", "application/vnd.oai.openapi+json")
	for _, statement := range []string{`
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
SELECT 'project',$1,'historical-results','arch-'||i,'archive-'||i,
 (SELECT version_id FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1 AND namespace='historical-results' AND name='typed-archive' AND revision=$2)
FROM generate_series(1,2000) i`, `
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes,material_rank)
SELECT scope_kind,scope_id,namespace,name,revision,true,1,$3,1
FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1
 AND namespace='historical-results' AND revision<>$2`} {
		args := []any{id, *archive.Ref.Revision}
		if strings.Contains(statement, "$3") {
			args = append(args, archive.Size)
		}
		if _, err := pool.Exec(ctx, statement, args...); err != nil {
			t.Fatal(err)
		}
	}
	query := BindingPageQuery{MaterialOrder: true, ExcludeNamespacePrefix: "audit-", HiddenNamespacePrefix: "historical-", Limit: 3}
	var listed []string
	for {
		page, err := project.ListMetadata(ctx, query)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range page {
			listed = append(listed, item.Ref.Namespace+"/"+item.Ref.Name)
		}
		if len(page) < query.Limit {
			break
		}
		last := page[len(page)-1]
		query.AfterNamespace, query.AfterName = last.Ref.Namespace, last.Ref.Name
		query.AfterMaterialRank = last.MaterialRank
	}
	want := []string{"sources/code", "z-imports/git", "z-output/api", "engagement/scope", "likec4/model", "docs/guide", "diffs/patch"}
	if len(listed) != 1207 || !reflect.DeepEqual(listed[:7], want) {
		t.Fatalf("priority page prefix=%v total=%d", listed[:min(7, len(listed))], len(listed))
	}
	seen := map[string]bool{}
	for _, ref := range listed {
		if seen[ref] {
			t.Fatalf("duplicate %s", ref)
		}
		seen[ref] = true
	}
	generic, err := project.ListMetadata(ctx, BindingPageQuery{ExcludeNamespacePrefix: "audit-", Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if generic[0].Ref.Namespace != "a-generated" {
		t.Fatal("default order changed")
	}
	namespace := "historical-results"
	archived, err := project.ListMetadata(ctx, BindingPageQuery{MaterialOrder: true, Namespace: &namespace, ExcludeNamespacePrefix: "audit-", Limit: 3})
	if err != nil || len(archived) != 3 || archived[0].MaterialRank != 1 {
		t.Fatalf("explicit ordered archive page: %+v %v", archived, err)
	}
	if _, err := project.Read(ctx, archived[0].Ref); err != nil {
		t.Fatal(err)
	}

	// Changing the current format must replace the indexed rank atomically.
	original := write("z-output", "mutable", "application/json")
	if _, err := project.Write(ctx, ArtifactRef{Namespace: "z-output", Name: "mutable"}, Payload{MediaType: "application/vnd.oai.openapi+yaml", Data: []byte("api")}, original.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	first, err := project.ListMetadata(ctx, BindingPageQuery{MaterialOrder: true, ExcludeNamespacePrefix: "audit-", HiddenNamespacePrefix: "historical-", Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if first[3].Ref.Name != "mutable" || first[3].MaterialRank != 1 {
		t.Fatalf("new revision rank: %+v", first)
	}

	// Both page variants must seek their index before metadata hydration.
	for i := 0; i < 120; i++ {
		write("z-output", fmt.Sprintf("other-%03d", i), "application/json")
	}
	for _, test := range []struct{ sql, namespace, index string }{
		{artifactpostgres.ListProjectMaterialMetadataSQL, "", "artifact_bindings_project_material_order_idx"},
		{artifactpostgres.ListProjectNamespaceMaterialMetadataSQL, "z-output", "artifact_bindings_project_material_namespace_order_idx"},
	} {
		var ns any
		if test.namespace != "" {
			ns = test.namespace
		}
		if _, err := pool.Exec(ctx, "ANALYZE artifact_bindings"); err != nil {
			t.Fatal(err)
		}
		var raw []byte
		if err := pool.QueryRow(ctx, "EXPLAIN (ANALYZE,FORMAT JSON) "+test.sql, id, ns, 0, "", "", 4).Scan(&raw); err != nil {
			t.Fatal(err)
		}
		var plans []struct{ Plan explainNode }
		if err := json.Unmarshal(raw, &plans); err != nil {
			t.Fatal(err)
		}
		scans := plans[0].Plan.scansOf("artifact_bindings")
		if len(scans) != 1 || scans[0].IndexName != test.index || scans[0].ActualRows > 4 || scans[0].RowsRemovedByFilter != 0 {
			t.Fatalf("unbounded material page %+v\n%s", scans, raw)
		}
	}

	// Exact input and output forks preserve Git provenance, while an explicit
	// replacement derives its rank from the new version and target namespace.
	projectID := id
	runs := runstore.NewPostgresStore(pool, storage)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "material-fork-run", OwnerID: "user-1", ProjectID: &projectID, WorkflowName: "copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	runScope, _ := RunScope("material-fork-run")
	input, err := service.Fork(ctx, Fork{Source: scope, SourceRef: imported.Ref, Target: runScope, TargetRef: ArtifactRef{Namespace: "inputs", Name: "code"}, Kind: LineageInputFork})
	if err != nil {
		t.Fatal(err)
	}
	published, err := service.Fork(ctx, Fork{Source: runScope, SourceRef: input.TargetRef, Target: scope, TargetRef: ArtifactRef{Namespace: "z-published", Name: "code"}, Kind: LineageProjectOutputPublish})
	if err != nil {
		t.Fatal(err)
	}
	var rank int
	if err := pool.QueryRow(ctx, "SELECT material_rank FROM artifact_bindings WHERE scope_id=$1 AND namespace='z-published'", id).Scan(&rank); err != nil || rank != 0 {
		t.Fatalf("fork rank=%d: %v", rank, err)
	}
	runStore, _ := service.Run("material-fork-run")
	replacement, err := runStore.Write(ctx, ArtifactRef{Namespace: "reports", Name: "report"}, Payload{MediaType: "text/markdown", Data: []byte("replacement")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Exercise the shared storage replacement statement directly; the Service
	// separately restricts its public caller to Audit-to-Project publication.
	if _, err := storage.Repository(pool).Fork(ctx, Fork{Source: runScope, SourceRef: replacement.Ref, Target: scope, TargetRef: ArtifactRef{Namespace: "z-published", Name: "code"}, Kind: LineageProjectOutputPublish, ExpectedRevision: published.TargetRef.Revision}); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT material_rank FROM artifact_bindings WHERE scope_id=$1 AND namespace='z-published'", id).Scan(&rank); err != nil || rank != 4 {
		t.Fatalf("replacement rank=%d: %v", rank, err)
	}
}
