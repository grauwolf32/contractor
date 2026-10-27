package app

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresFindingChecksDataMigrationCLIUsesExactInlineAndFilesystemSources(t *testing.T) {
	base := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if base == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	databaseURL := isolatedStartupDatabase(t, ctx, base)
	pool, err := persistencepostgres.OpenPool(ctx, databaseURL, persistencepostgres.PoolOptions{MaxConnections: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, err = persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	digest := auditdomain.DigestBytes([]byte("fixture"))
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project-checks-cli", OwnerID: "owner-checks-cli", Name: "CLI fixture", Kind: projectstore.KindProject, IdempotencyKey: "project", RequestDigest: digest})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err := ownerstore.New(pool).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "audit-checks-cli", OwnerID: project.OwnerID, ProjectID: project.ProjectID, Profile: auditstore.ProfileIdentity{Name: "fixture", Version: "1", Digest: digest}, ProfileSnapshot: json.RawMessage(`{"fixture":true}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: "audit", RequestDigest: digest})
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir()
	_, filesystem, err := artifacts.OpenBlobStore(ctx, artifacts.BlobFilesystem, path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if closer, ok := filesystem.(io.Closer); ok {
		defer closer.Close()
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	scope, _ := artifacts.ProjectScope(project.ProjectID)
	for index, kind := range []string{"inline", "filesystem"} {
		sourceCtx := ctx
		if index == 1 {
			sourceCtx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(filesystem, logger))
		}
		doc := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: kind, Title: "CLI preservation", Description: "Original bytes survive explicit migration.", Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{{Method: "static-trace", Objective: "Trace exact Кириллица <>& bytes"}}, SeveritySuggestion: "medium", Limitations: []string{}}
		data, e := auditdomain.EncodeFindingProposal(doc)
		if e != nil {
			t.Fatal(e)
		}
		written, e := artifacts.NewService(artifacts.NewPostgresRepository(pool)).CreateFrozen(sourceCtx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: kind}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: data})
		if e != nil {
			t.Fatal(e)
		}
		exact := contracts.ExactContent(written.Ref, written.MediaType, data)
		ref, _ := json.Marshal(exact.Ref)
		descriptor, _ := json.Marshal(exact)
		_, e = pool.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection)
VALUES($1,$1,$1,$1,$1,$1,'worker',$1,$1,$1,$2,'gone-source',$3,$4,'fixture','1','contractor/v1alpha1','{"name":"fixture","version":"1"}',$2,$5::jsonb,$6,'application/json',$7,'[]','{"description":"Original bytes survive explicit migration.","limitations":[],"preconditions":[],"severity_suggestion":"medium","standard_refs":[],"subject":null,"title":"CLI preservation"}')`, "receipt-cli-"+kind, digest, project.OwnerID, project.ProjectID, ref, exact.Digest, exact.SizeBytes)
		if e != nil {
			t.Fatal(e)
		}
		if _, e = pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES($1,$2,$3,$4::jsonb,'[]')`, "receipt-cli-"+kind, audit.AuditID, project.ProjectID, descriptor); e != nil {
			t.Fatal(e)
		}
	}
	var backends int
	if err = pool.QueryRow(ctx, `SELECT count(DISTINCT backend) FROM artifact_blobs`).Scan(&backends); err != nil || backends != 2 {
		t.Fatalf("physical source backends=%d,%v", backends, err)
	}
	var before string
	if err = pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'] ORDER BY receipt_id)::text FROM finding_proposal_receipts AS receipt`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	env := func(key string) string {
		switch key {
		case "CONTRACTOR_DATABASE_URL":
			return databaseURL
		case "CONTRACTOR_ARTIFACT_BLOB_PATH":
			return path
		}
		return ""
	}
	for range 2 {
		if err = RunCLI(ctx, []string{"data-migrate", "finding-checks", "--batch-size=1"}, env, logger); err != nil {
			t.Fatal(err)
		}
	}
	var after string
	var count int
	if err = pool.QueryRow(ctx, `SELECT jsonb_agg(to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'] ORDER BY receipt_id)::text,count(*) FILTER(WHERE checks_projection IS NOT NULL) FROM finding_proposal_receipts AS receipt`).Scan(&after, &count); err != nil || after != before || count != 2 {
		t.Fatalf("preservation=%v indexed=%d error=%v", after == before, count, err)
	}
}
