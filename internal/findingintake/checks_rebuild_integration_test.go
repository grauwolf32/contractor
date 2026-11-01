//go:build integration

package findingintake

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresFindingChecksIndexIntakeRebuildAndRetention(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedFindingPool(t, ctx)
			if backend == "filesystem" {
				files, err := artifacts.OpenFilesystemBlobStore(ctx, t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))
			}
			const owner, project, run = "owner", "delete-project", "run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project,
				OwnerID: owner, Kind: projectstore.KindProject, Name: "Check indexes", IdempotencyKey: "project",
				RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
				t.Fatal(err)
			}
			grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
			service, _ := New(pool)
			input := testSubmission("checks", "checks", nil)
			input.Proposal.ProposedChecks = []auditdomain.ProposedCheck{
				{Objective: "Verify UTF-8 世界, braces }{ and a quoted \"proposed_checks\" key.", Method: "static-trace"},
				{Objective: "Preserve an ordinary unsupported method.", Method: "operator-custom-method"},
			}
			receipt, _, err := service.Submit(ctx, grant, input)
			if err != nil {
				t.Fatal(err)
			}
			readIndex := func() []byte {
				t.Helper()
				var data []byte
				if err := pool.QueryRow(ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&data); err != nil {
					t.Fatal(err)
				}
				return data
			}
			initial := readIndex()
			projection, err := auditdomain.DecodeFindingChecksProjection(initial)
			if err != nil || len(projection.Checks) != 2 || projection.Checks[0].Status != "registered" || projection.Checks[1].Status != "unsupported" {
				t.Fatalf("trusted index: %+v %v", projection, err)
			}
			store, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(run)
			source, err := store.ReadExact(ctx, receipt.Proposal)
			if err != nil {
				t.Fatal(err)
			}
			check, plan, err := auditdomain.ExtractFindingCheck(source.Payload.Data, projection.Checks[0])
			if err != nil || check != input.Proposal.ProposedChecks[0] || plan.Method != check.Method {
				t.Fatalf("selected exact fragment: %+v %+v %v", check, plan, err)
			}
			if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection=NULL WHERE receipt_id=$1`, receipt.ReceiptID); err == nil {
				t.Fatal("published private index was mutable")
			}
			// Model the preceding schema without changing the real receipt,
			// proposal or trusted origin used by the explicit rebuild operation.
			if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection=NULL WHERE receipt_id=$1`, receipt.ReceiptID); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
				t.Fatal(err)
			}
			var before, after string
			if err := pool.QueryRow(ctx, `SELECT (to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'])::text FROM finding_proposal_receipts receipt WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&before); err != nil {
				t.Fatal(err)
			}
			f := deletionImportFixture{ctx: ctx, pool: pool, intake: service, receiptID: receipt.ReceiptID,
				request: ImportRequest{OwnerID: owner, RunID: run, Proposal: receipt.Proposal.Ref}}
			createFixtureAudit(t, f, "audit")
			request := f.request
			request.AuditID = "audit"
			if _, _, err := service.ImportIntoAudit(ctx, request); err != nil {
				t.Fatal(err)
			}
			runs := runstore.NewPostgresStore(pool)
			if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
				t.Fatal(err)
			}
			if _, err := runs.TransitionRun(ctx, run, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
				t.Fatal(err)
			}
			if err := deleteRun(ctx, pool, owner, run); err != nil {
				t.Fatal(err)
			}
			// Capture preservation after the expected retention transition.
			if err := pool.QueryRow(ctx, `SELECT (to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'])::text FROM finding_proposal_receipts receipt WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&before); err != nil {
				t.Fatal(err)
			}
			report, err := service.RebuildChecksProjectionBatch(ctx, 128)
			if err != nil || report.Rebuilt != 1 || report.More || !bytes.Equal(initial, readIndex()) {
				t.Fatalf("held source rebuild: %+v %v", report, err)
			}
			if err := pool.QueryRow(ctx, `SELECT (to_jsonb(receipt)-ARRAY['checks_projection','checks_projection_bytes','checks_projection_count'])::text FROM finding_proposal_receipts receipt WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&after); err != nil || before != after {
				t.Fatalf("data migration changed receipt provenance: %v", err)
			}
			report, err = service.RebuildChecksProjectionBatch(ctx, 128)
			if err != nil || report.Rebuilt != 0 || report.More {
				t.Fatalf("rebuild replay: %+v %v", report, err)
			}
			if _, err := service.RebuildChecksProjectionBatch(ctx, 129); !errors.Is(err, ErrInvalid) {
				t.Fatal("unbounded rebuild accepted", err)
			}
		})
	}
}

func TestPostgresFindingChecksRebuildMissingOrCorruptFilesystemBytes(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	path := t.TempDir()
	files, err := artifacts.OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project", OwnerID: "owner",
		Kind: projectstore.KindProject, Name: "Missing bytes", IdempotencyKey: "project", RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, "owner", "project", "run")
	service, _ := New(pool)
	input := testSubmission("missing", "missing", nil)
	input.Proposal.ProposedChecks = []auditdomain.ProposedCheck{{Objective: "Verify", Method: "verify"}}
	receipt, _, err := service.Submit(ctx, grant, input)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET checks_projection=NULL`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`); err != nil {
		t.Fatal(err)
	}
	var key string
	if err := pool.QueryRow(ctx, `SELECT object_key FROM artifact_blobs WHERE sha256=decode(substring($1 from 8),'hex')`, receipt.Proposal.Digest).Scan(&key); err != nil {
		t.Fatal(err)
	}
	filename := filepath.Join(path, key)
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filename, filename+".saved"); err != nil {
		t.Fatal(err)
	}
	if _, err := service.RebuildChecksProjectionBatch(ctx, 128); !errors.Is(err, artifacts.ErrBlobMissing) {
		t.Fatalf("missing source was hidden: %v", err)
	}
	if err := os.WriteFile(filename, bytes.Repeat([]byte{'x'}, len(content)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.RebuildChecksProjectionBatch(ctx, 128); !errors.Is(err, artifacts.ErrArtifactIntegrity) {
		t.Fatalf("corrupt source was hidden: %v", err)
	}
	var pending bool
	if err := pool.QueryRow(ctx, `SELECT checks_projection IS NULL FROM finding_proposal_receipts WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&pending); err != nil || !pending {
		t.Fatal("failed rebuild published authority", err)
	}
	if err := os.WriteFile(filename, content, 0o600); err != nil {
		t.Fatal(err)
	}
	if report, err := service.RebuildChecksProjectionBatch(ctx, 128); err != nil || report.Rebuilt != 1 {
		t.Fatalf("restored exact source did not recover: %+v %v", report, err)
	}
	var encoded []byte
	if err := pool.QueryRow(ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id=$1`, receipt.ReceiptID).Scan(&encoded); err != nil {
		t.Fatal(err)
	}
	var projection auditdomain.FindingChecksProjection
	if strictjson.Decode(encoded, &projection) != nil || len(projection.Checks) != 1 {
		t.Fatal("restored projection is invalid")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := service.RebuildChecksProjectionBatch(cancelled, 1); !errors.Is(err, context.Canceled) {
		t.Fatal("cancel ignored", err)
	}
}
