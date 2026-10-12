package artifacts_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// The generic Project listing hides every audit- namespace and pages the
// other bindings in order, including names that a linguistic collation sorts
// between Audit namespaces, and it reads them from the listing index without
// touching an Audit-managed binding (ART-R26, ART-R115).
func TestPostgresIntegrationProjectListingSkipsAuditNamespacesByIndex(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	const projectID = "listing-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Listing", IdempotencyKey: "create-listing-project",
		RequestDigest: "sha256:" + strings.Repeat("d", 64),
	}); err != nil {
		t.Fatal(err)
	}
	project, _ := NewService(NewPostgresRepository(pool)).Project(projectID)
	visible := []ArtifactRef{
		{Namespace: "audit0", Name: "first"}, {Namespace: "auditab", Name: "second"},
		{Namespace: "notes", Name: "third"}, {Namespace: "notes", Name: "fourth"},
		{Namespace: "zeta", Name: "fifth"},
	}
	for _, ref := range visible {
		if _, err := project.Write(ctx, ref, Payload{MediaType: "text/plain", Data: []byte(ref.Name)}, nil); err != nil {
			t.Fatalf("write %s/%s: %v", ref.Namespace, ref.Name, err)
		}
	}
	// Three Audit namespaces of 2,000 bindings each, named as the Server
	// derives them: audit- and 64 hex characters.
	for _, statement := range []string{`
INSERT INTO artifact_blobs (sha256, payload, size_bytes, backend)
SELECT sha256(convert_to('audit-' || i, 'UTF8')), convert_to('audit-' || i, 'UTF8'),
       octet_length(convert_to('audit-' || i, 'UTF8')), 'postgresql'
FROM generate_series(1, 6000) AS i`, `
INSERT INTO artifact_versions (version_id, blob_sha256, media_type)
SELECT 'audit-version-' || i, sha256(convert_to('audit-' || i, 'UTF8')), 'application/json'
FROM generate_series(1, 6000) AS i`, `
INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
SELECT 'project', $1, 'audit-' || encode(sha256(convert_to('audit ' || (i % 3), 'UTF8')), 'hex'),
       'item-' || i, 'rev-' || i, 'audit-version-' || i
FROM generate_series(1, 6000) AS i`, `
INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
SELECT scope_kind, scope_id, namespace, name, revision
FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = $1 AND namespace LIKE 'audit-%'`} {
		var arguments []any
		if strings.Contains(statement, "$1") {
			arguments = append(arguments, projectID)
		}
		if _, err := pool.Exec(ctx, statement, arguments...); err != nil {
			t.Fatalf("seed Audit bindings: %v", err)
		}
	}
	if _, err := pool.Exec(ctx, `ANALYZE artifact_bindings`); err != nil {
		t.Fatal(err)
	}

	// Page through the whole listing in one transaction, so its statistics
	// count every binding row the pages read.
	var listed []string
	var bindingReads int64
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		pages, _ := NewService(NewPostgresRepository(tx)).Project(projectID)
		query := BindingPageQuery{ExcludeNamespacePrefix: "audit-", Limit: 2}
		for {
			page, err := pages.ListMetadata(ctx, query)
			if err != nil {
				return err
			}
			for _, metadata := range page {
				listed = append(listed, metadata.Ref.Namespace+"/"+metadata.Ref.Name)
			}
			if len(page) < query.Limit {
				break
			}
			last := page[len(page)-1].Ref
			query.AfterNamespace, query.AfterName = last.Namespace, last.Name
		}
		return tx.QueryRow(ctx, `
SELECT coalesce(sum(seq_tup_read + idx_tup_fetch), 0)
FROM pg_stat_xact_user_tables WHERE relname = 'artifact_bindings'`).Scan(&bindingReads)
	}); err != nil {
		t.Fatal(err)
	}
	want := "audit0/first auditab/second notes/fourth notes/third zeta/fifth"
	if got := strings.Join(listed, " "); got != want {
		t.Fatalf("Project listing = %q, want %q", got, want)
	}
	// Filtering on the primary key reads all 6,000 Audit bindings once.
	if bindingReads > 20 {
		t.Fatalf("Project listing read %d binding rows for %d listed bindings", bindingReads, len(visible))
	}

	var plan []byte
	if err := pool.QueryRow(ctx, "EXPLAIN (ANALYZE, FORMAT JSON) "+ListProjectMetadataSQL,
		projectID, nil, nil, "", "", 3, "",
	).Scan(&plan); err != nil {
		t.Fatal(err)
	}
	var plans []struct {
		Plan explainNode `json:"Plan"`
	}
	if err := json.Unmarshal(plan, &plans); err != nil || len(plans) != 1 {
		t.Fatalf("decode plan %s: %v", plan, err)
	}
	scans := plans[0].Plan.scansOf("artifact_bindings")
	if len(scans) != 1 || scans[0].IndexName != "artifact_bindings_project_listing_idx" ||
		scans[0].RowsRemovedByFilter != 0 || scans[0].ActualRows > 3 {
		t.Fatalf("Project listing reads artifact_bindings with %+v, want one listing index scan of at most 3 rows\n%s", scans, plan)
	}
}

// explainNode is the part of an EXPLAIN (FORMAT JSON) plan node the listing
// test inspects.
type explainNode struct {
	NodeType            string        `json:"Node Type"`
	RelationName        string        `json:"Relation Name"`
	IndexName           string        `json:"Index Name"`
	ActualRows          float64       `json:"Actual Rows"`
	RowsRemovedByFilter float64       `json:"Rows Removed by Filter"`
	Plans               []explainNode `json:"Plans"`
}

// scansOf returns the plan nodes that read relation.
func (node explainNode) scansOf(relation string) []explainNode {
	var result []explainNode
	if node.RelationName == relation {
		result = append(result, node)
	}
	for _, child := range node.Plans {
		result = append(result, child.scansOf(relation)...)
	}
	return result
}
