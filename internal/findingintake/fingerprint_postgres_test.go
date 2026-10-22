//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresFindingFingerprintIntakeReplayAndRetention(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	base := isolatedFindingPool(t, ctx)
	config := base.Config()
	config.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	const owner, project, run = "fingerprint-owner", "delete-project", "fingerprint-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Fingerprint project", IdempotencyKey: "fingerprint", RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	var refs []contracts.ArtifactRef
	grant := quotaSubmissionGrantWithInputs(t, ctx, pool, owner, project, run, func() map[string]runstore.PinnedContextArtifact {
		refs = writeFingerprintInputs(t, ctx, pool, run)
		return map[string]runstore.PinnedContextArtifact{"source": {Artifact: &refs[0]}, "report": {Artifact: &refs[1]}}
	})
	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	input := testSubmission("invocation-a", "finding-a", refs)
	line := int64(3)
	input.Proposal.Locations = []auditdomain.FindingLocation{{File: "src/a.py", Line: &line}}
	first, replayed, err := service.Submit(ctx, grant, input)
	if err != nil || replayed || first.Fingerprint.Availability != "available" {
		t.Fatalf("native intake=%+v replay=%v err=%v", first, replayed, err)
	}
	// An independent call remains a separate observation, including its prose.
	secondInput := testSubmission("invocation-b", "finding-b", refs)
	secondInput.Proposal.Locations = input.Proposal.Locations
	secondInput.Proposal.Title = "Independent audit observation"
	second, _, err := service.Submit(ctx, grant, secondInput)
	if err != nil || second.ReceiptID == first.ReceiptID || second.Fingerprint.Value != first.Fingerprint.Value {
		t.Fatalf("independent observation=%+v: %v", second, err)
	}
	wire, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	var receiptFields map[string]json.RawMessage
	if err := json.Unmarshal(wire, &receiptFields); err != nil {
		t.Fatal(err)
	}
	if _, exposed := receiptFields["fingerprint"]; exposed || strings.Contains(string(wire), first.Fingerprint.Value) {
		t.Fatalf("derived metadata leaked into receipt wire: %s", wire)
	}
	changed := input
	changed.Proposal.Title = "Changed replay"
	if _, _, err := service.Submit(ctx, grant, changed); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed replay=%v", err)
	}
	revision := "1"
	foreign := testSubmission("foreign", "finding-foreign", []contracts.ArtifactRef{{Namespace: "foreign", Name: "report", Revision: &revision}})
	foreign.Proposal.Locations = input.Proposal.Locations
	if _, _, err := service.Submit(ctx, grant, foreign); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("foreign ref=%v", err)
	}
	cancelled, stop := context.WithCancel(ctx)
	stop()
	if _, _, err := service.Submit(cancelled, grant, testSubmission("cancelled", "cancelled", nil)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled=%v", err)
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts").Scan(&count); err != nil || count != 2 {
		t.Fatalf("receipts=%d: %v", count, err)
	}
	f := deletionImportFixture{ctx: ctx, pool: pool, intake: service, receiptID: first.ReceiptID, request: ImportRequest{OwnerID: owner, RunID: run, Proposal: first.Proposal.Ref}}
	for _, id := range []string{"fingerprint-audit-a", "fingerprint-audit-b"} {
		createFixtureAudit(t, f, id)
		req := f.request
		req.AuditID = id
		if _, _, err := service.ImportIntoAudit(ctx, req); err != nil {
			t.Fatal(err)
		}
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
	// Replay uses its stored projection even after every original Run input is gone.
	replay, ok, err := service.Submit(ctx, grant, input)
	if err != nil || !ok || replay.ReceiptID != first.ReceiptID || replay.Fingerprint.Value != first.Fingerprint.Value {
		t.Fatalf("replay after source deletion=%+v %v %v", replay, ok, err)
	}
	for _, id := range []string{"fingerprint-audit-a", "fingerprint-audit-b"} {
		retained, err := service.GetAuditReceipt(ctx, owner, id, first.ReceiptID)
		if err != nil || retained.Fingerprint.Value != first.Fingerprint.Value || retained.Retention != RetentionAuditHeld {
			t.Fatalf("retained projection=%+v %v", retained, err)
		}
		if _, err := service.GetAuditReceipt(ctx, "foreign-owner", id, first.ReceiptID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("owner isolation=%v", err)
		}
		audits := auditstoretest.NewStore(pool).WithPurgeDependencies(DeleteAuditHoldsBatch)
		audit, err := audits.Get(ctx, owner, id)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := audits.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: owner, AuditID: id, ExpectedRevision: audit.Revision, IdempotencyKey: "purge-" + id, RequestDigest: auditdomain.DigestBytes([]byte(id))}); err != nil {
			t.Fatal(err)
		}
		claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "fingerprint-purge", Lease: time.Minute, Limit: 1})
		if err != nil || len(claims) != 1 {
			t.Fatalf("claim=%+v %v", claims, err)
		}
		if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(id)); err != nil {
			t.Fatal(err)
		}
	}
	discarded, err := readReceiptByID(ctx, pool, first.ReceiptID)
	if err != nil || discarded.Fingerprint.Reason != "discarded" || discarded.Fingerprint.Value != "" {
		t.Fatalf("discarded sensitive projection=%+v %v", discarded.Fingerprint, err)
	}
	unimported, err := readReceiptByID(ctx, pool, second.ReceiptID)
	if err != nil || unimported.Fingerprint.Reason != "discarded" {
		t.Fatalf("unimported discard=%+v %v", unimported.Fingerprint, err)
	}
}

func writeFingerprintInputs(t *testing.T, ctx context.Context, pool *pgxpool.Pool, run string) []contracts.ArtifactRef {
	t.Helper()
	const report = `{"version":"2.1.0","runs":[{"tool":{"driver":{"name":"scanner","version":"1"}},"versionControlProvenance":[{"repositoryUri":"https://example.test:443/repo.git","revisionId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"results":[{"ruleId":"rule/sql","fingerprints":{"native/v1":"issue-a"},"locations":[{"physicalLocation":{"artifactLocation":{"uri":"src/a.py"},"region":{"startLine":3}}}]}]}]}`
	var refs []contracts.ArtifactRef
	for index, payload := range []artifacts.Payload{{MediaType: "application/zip", Data: []byte("exact source")}, {MediaType: "application/sarif+json", Data: []byte(report)}} {
		prepared, err := artifacts.PreparePayload(ctx, payload)
		if err != nil {
			t.Fatal(err)
		}
		err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repo := artifacts.NewPostgresTxRepository(tx)
			store, err := artifacts.NewService(repo).Run(run)
			if err != nil {
				return err
			}
			name := "source"
			if index == 1 {
				name = "report"
			}
			written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: name}, prepared, nil)
			if err != nil {
				return err
			}
			refs = append(refs, written.Ref)
			if index == 0 {
				scope, err := artifacts.RunScope(run)
				if err != nil {
					return err
				}
				return repo.RecordGitSource(ctx, scope, written.Ref, artifacts.GitSource{RepositoryURL: "https://example.test:443/repo.git", ResolvedCommit: strings.Repeat("a", 40), ImportedAt: time.Now().UTC()})
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return refs
}
