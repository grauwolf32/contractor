package auditservice

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func createCorrelationAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, project, audit string) {
	t.Helper()
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Correlation", IdempotencyKey: project, RequestDigest: serviceTestDigest(project)})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = auditstoretest.NewStore(pool, artifactpgtest.NewStorage()).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: audit, OwnerID: owner, ProjectID: project, Profile: auditstore.ProfileIdentity{Name: "correlation", Version: "1", Digest: serviceTestDigest("profile")}, ProfileSnapshot: json.RawMessage(`{"interaction":{}}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2, MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: audit, RequestDigest: serviceTestDigest(audit)})
	if err != nil {
		t.Fatal(err)
	}
}

func correlationFingerprint(t *testing.T) auditdomain.FindingFingerprint {
	t.Helper()
	revision := "1"
	source := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}, "application/zip", []byte("source"))
	report := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "report", Revision: &revision}, "application/sarif+json", []byte("report"))
	value := auditdomain.FindingFingerprint{Schema: auditdomain.FindingFingerprintSchema, Availability: "available", Algorithm: "contractor/producer-issue/v1", Value: strings.Repeat("a", 64), Adapter: "pinned-sarif-native/v1", RuleID: "rule/test", Repository: "https://example.test/repo.git", Commit: strings.Repeat("a", 40), SourceDigest: source.Digest, Source: &source, Report: &report, RecordLocator: "/runs/0/results/0"}
	if _, err := auditdomain.EncodeFindingFingerprint(value); err != nil {
		t.Fatal(err)
	}
	return value
}

type correlationQueryTrace struct {
	sql  string
	args []any
}

func (q *correlationQueryTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.sql, q.args = data.SQL, data.Args
	return ctx
}
func (*correlationQueryTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
