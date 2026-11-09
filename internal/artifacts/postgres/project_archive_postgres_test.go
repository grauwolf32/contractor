package postgres_test

import (
	"context"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/policy"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func TestPostgresIntegrationProjectArchiveDefaultPages(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	const projectID = "archive-listing-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Archive listing", IdempotencyKey: "archive-listing",
		RequestDigest: "sha256:" + strings.Repeat("d", 64),
	}); err != nil {
		t.Fatal(err)
	}
	storage := artifactpgtest.NewStorage()
	project, _ := NewService(storage.Repository(pool)).Project(projectID)
	for _, name := range []string{"one", "two", "three"} {
		if _, err := project.Write(ctx, ArtifactRef{Namespace: "notes", Name: name}, Payload{MediaType: "text/plain", Data: []byte("shared")}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// Archives outnumber current materials by three orders of magnitude.
	for _, statement := range []string{`
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
SELECT 'project',$1, CASE i%4 WHEN 0 THEN 'historical-audits'
 WHEN 1 THEN 'historical-materials' WHEN 2 THEN 'historical-results'
 ELSE 'audit-cutover-materials' END, 'entry-'||i,'archive-rev-'||i,
 (SELECT version_id FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1 LIMIT 1)
FROM generate_series(1,8000) i`, `
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
SELECT scope_kind,scope_id,namespace,name,revision,true,1,6 FROM artifact_binding_revisions
WHERE scope_kind='project' AND scope_id=$1 AND namespace<>'notes'`} {
		if _, err := pool.Exec(ctx, statement, projectID); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `ANALYZE artifact_bindings`); err != nil {
		t.Fatal(err)
	}
	var listed []string
	var reads int64
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		pages, _ := NewService(storage.Repository(tx)).Project(projectID)
		query := BindingPageQuery{
			ExcludeNamespacePrefix: policy.AuditManagedProjectNamespacePrefix,
			HiddenNamespacePrefix:  policy.HistoricalProjectNamespacePrefix,
			Limit:                  2,
		}
		for {
			items, err := pages.ListMetadata(ctx, query)
			if err != nil {
				return err
			}
			for _, item := range items {
				listed = append(listed, item.Ref.Name)
			}
			if len(items) < query.Limit {
				break
			}
			last := items[len(items)-1].Ref
			query.AfterNamespace, query.AfterName = last.Namespace, last.Name
		}
		return tx.QueryRow(ctx, `SELECT coalesce(sum(seq_tup_read+idx_tup_fetch),0)
FROM pg_stat_xact_user_tables WHERE relname='artifact_bindings'`).Scan(&reads)
	}); err != nil {
		t.Fatal(err)
	}
	if strings.Join(listed, " ") != "one three two" || reads > 12 {
		t.Fatalf("default materials=%v, binding reads=%d", listed, reads)
	}
	for _, namespace := range []string{"historical-audits", "historical-materials", "historical-results"} {
		items, err := project.ListMetadata(ctx, BindingPageQuery{Namespace: &namespace, ExcludeNamespacePrefix: policy.AuditManagedProjectNamespacePrefix, Limit: 2})
		if err != nil || len(items) != 2 {
			t.Fatalf("explicit %s: %v, %v", namespace, items, err)
		}
		if _, err := project.Read(ctx, items[0].Ref); err != nil {
			t.Fatal(err)
		}
	}
}

func TestProjectMaterialsSQLMatchesArchivePolicy(t *testing.T) {
	for _, predicate := range []string{
		"namespace NOT LIKE '" + policy.AuditManagedProjectNamespacePrefix + "%'",
		"namespace NOT LIKE '" + policy.HistoricalProjectNamespacePrefix + "%'",
	} {
		if !strings.Contains(artifactpostgres.ListProjectMaterialsSQL, predicate) {
			t.Fatalf("missing %s", predicate)
		}
	}
}
