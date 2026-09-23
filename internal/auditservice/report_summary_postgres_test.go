package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// These regressions share the current report-acceptance fixture. Historical
// shapes are injected only into that disposable fixture, then restored so the
// ordinary current approval, expiry, rejection and post-approval checks run.
func testLegacyReportWriteRejection(t *testing.T, ctx context.Context, pool *pgxpool.Pool, current auditstore.ProposeReportParams) {
	t.Helper()
	legacy := current
	legacy.Summary.Artifact.MediaType = "text/plain"
	store := auditstore.NewPostgresStore(pool)
	for name, operation := range map[string]func() error{
		"propose-plain-summary": func() error { _, _, err := store.ProposeReport(ctx, legacy); return err },
		"commit-plain-summary":  func() error { _, err := store.CommitReport(ctx, auditstore.CommitReportParams(legacy)); return err },
	} {
		t.Run(name, func(t *testing.T) {
			assertReportOperationReadOnly(t, ctx, pool, func() {
				if err := operation(); !errors.Is(err, auditstore.ErrInvalid) {
					t.Fatalf("legacy write = %v, want ErrInvalid", err)
				}
			})
		})
	}
}

func testLegacyReportCandidateRejection(t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *Service, owner string, current auditstore.ProposeReportParams, candidate auditstore.ReportCandidate) {
	t.Helper()
	var original []byte
	if err := pool.QueryRow(ctx, `SELECT summary_link FROM audit_report_candidates WHERE audit_id=$1`, current.Claim.AuditID).Scan(&original); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_report_candidates SET summary_link=jsonb_set(summary_link, '{artifact,mediaType}', '"text/plain"'::jsonb) WHERE audit_id=$1`, current.Claim.AuditID); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if _, err := pool.Exec(ctx, `UPDATE audit_report_candidates SET summary_link=$2::jsonb WHERE audit_id=$1`, current.Claim.AuditID, original); err != nil {
			t.Fatal(err)
		}
	}()
	store := auditstore.NewPostgresStore(pool)
	for name, operation := range map[string]func() error{
		"read-plain-candidate":    func() error { _, err := store.GetReportCandidate(ctx, current.Claim.AuditID); return err },
		"project-plain-candidate": func() error { _, err := service.GetReport(ctx, owner, current.Claim.AuditID); return err },
		"replay-plain-candidate":  func() error { _, _, err := store.ProposeReport(ctx, current); return err },
		"approve-plain-candidate": func() error {
			_, err := service.DecideActionReview(ctx, DecideActionReviewParams{
				OwnerID: owner, AuditID: current.Claim.AuditID, RequestID: candidate.RequestID,
				ExpectedRequestRevision: 1, DecisionID: "legacy-approval", Action: ReviewApprove,
				Rationale:      "Historical candidate must not be published.",
				IdempotencyKey: "legacy-approval", RequestDigest: serviceTestDigest("legacy-approval"),
			})
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			assertReportOperationReadOnly(t, ctx, pool, func() {
				if err := operation(); err == nil {
					t.Fatal("historical report candidate was accepted")
				}
			})
		})
	}
}

func testLegacyCommittedReportRejection(t *testing.T, ctx context.Context, pool *pgxpool.Pool, service *Service, owner, project string, current auditstore.ProposeReportParams) {
	t.Helper()
	before, err := service.GetReport(ctx, owner, current.Claim.AuditID)
	if err != nil || before.Status != ReportReady || before.Summary != "# Frozen report\n" || before.SummaryArtifact == nil || before.SummaryArtifact.MediaType != "text/markdown" {
		t.Fatalf("current committed Markdown report = %+v, %v", before, err)
	}
	if _, err := service.GetReport(ctx, "another-owner", current.Claim.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign report read = %v", err)
	}
	payload := []byte(before.Summary)
	written, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).WriteAuditArtifact(ctx, project,
		contracts.ArtifactRef{Namespace: "audit-report-review", Name: "historical-report.txt"},
		artifacts.Payload{MediaType: "text/plain", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	plain := auditstore.ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: "text/plain", SizeBytes: written.Size}
	mislabeled := plain
	mislabeled.MediaType = "text/markdown"
	install := func(artifact auditstore.ExactArtifact) {
		t.Helper()
		ref, err := json.Marshal(artifact.Ref)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `UPDATE audit_artifact_links SET artifact_ref=$2::jsonb,artifact_digest=$3,media_type=$4,size_bytes=$5 WHERE audit_id=$1 AND logical_key='report/summary'`,
			current.Claim.AuditID, ref, artifact.Digest, artifact.MediaType, artifact.SizeBytes); err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		name     string
		artifact auditstore.ExactArtifact
	}{
		{"read-committed-plain-summary", plain},
		{"read-plain-artifact-labeled-markdown", mislabeled},
	} {
		t.Run(test.name, func(t *testing.T) {
			install(test.artifact)
			defer install(current.Summary.Artifact)
			assertReportOperationReadOnly(t, ctx, pool, func() {
				if projection, err := service.GetReport(ctx, owner, current.Claim.AuditID); err == nil || projection.SummaryArtifact != nil || projection.Summary != "" {
					t.Fatalf("historical summary read = %+v, %v", projection, err)
				}
			})
		})
	}
	after, err := service.GetReport(ctx, owner, current.Claim.AuditID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("restored current report changed = %+v, %v", after, err)
	}
}

func assertReportOperationReadOnly(t *testing.T, ctx context.Context, pool *pgxpool.Pool, operation func()) {
	t.Helper()
	snapshot := func() map[string]string {
		t.Helper()
		state := map[string]string{}
		for _, table := range []string{
			"audits", "audit_rounds", "audit_items", "audit_coverage_rows", "audit_artifact_links",
			"audit_report_candidates", "audit_review_requests", "audit_review_decisions", "audit_events", "audit_idempotency",
			"artifact_scopes", "artifact_blobs", "artifact_versions", "artifact_binding_revisions", "artifact_bindings", "artifact_pins",
		} {
			var data string
			if err := pool.QueryRow(ctx, `SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY to_jsonb(row)::text), '[]'::jsonb)::text FROM `+pgx.Identifier{table}.Sanitize()+` AS row`).Scan(&data); err != nil {
				t.Fatal(err)
			}
			state[table] = data
		}
		return state
	}
	before := snapshot()
	for range 2 {
		operation()
		if !reflect.DeepEqual(before, snapshot()) {
			t.Fatal("rejected report operation changed retained state")
		}
	}
}
