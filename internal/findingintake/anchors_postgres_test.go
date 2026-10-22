//go:build integration

package findingintake

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresFindingAnchorRuntimeIntakeReplayAndRetention(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	base := isolatedFindingPool(t, ctx)
	cfg := base.Config()
	cfg.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var fixture struct {
		File      string
		Line      int64
		ZipBase64 string
		Derived   auditdomain.FindingAnchorDerivation
		Keys      []auditdomain.FindingAnchorKeys
	}
	data, err := os.ReadFile("testdata/anchor-runtime.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	zipData, err := base64.StdEncoding.DecodeString(fixture.ZipBase64)
	if err != nil {
		t.Fatal(err)
	}
	const owner, project, run = "anchor-owner", "delete-project", "anchor-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Anchors", IdempotencyKey: "anchors", RequestDigest: auditdomain.DigestBytes([]byte("anchors"))}); err != nil {
		t.Fatal(err)
	}
	var refs []contracts.ArtifactRef
	grant := quotaSubmissionGrantWithInputs(t, ctx, pool, owner, project, run, func() map[string]runstore.PinnedContextArtifact {
		imitation, _ := strictjson.Canonical(fixture.Derived)
		refs = writeFingerprintReportInputs(t, ctx, pool, run, imitation, zipData)
		return map[string]runstore.PinnedContextArtifact{"source": {Artifact: &refs[0]}}
	})
	fixture.Derived.SourceRef = &refs[0]
	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	submission := func(invocation string) Submission {
		input := testSubmission(invocation, "key-"+invocation, refs[:1])
		input.Proposal.Subject = nil
		input.Proposal.Locations = []auditdomain.FindingLocation{{File: fixture.File, Line: &fixture.Line}}
		input.Proposal.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-321"}}
		input.DerivedAnchors, err = strictjson.Canonical(fixture.Derived)
		if err != nil {
			t.Fatal(err)
		}
		return input
	}
	input := submission("first")
	first, replayed, err := service.Submit(ctx, grant, input)
	if err != nil || replayed || first.Anchors.Availability != "available" || len(first.Anchors.Keys) != 1 || first.Anchors.Keys[0] != fixture.Keys[0] {
		t.Fatalf("first=%+v replay=%v err=%v", first.Anchors, replayed, err)
	}
	secondInput := submission("second")
	secondInput.Proposal.Description = "Independent observation"
	secondInput.Proposal.SeveritySuggestion = "high"
	second, _, err := service.Submit(ctx, grant, secondInput)
	if err != nil || second.ReceiptID == first.ReceiptID || second.Anchors.Keys[0] != first.Anchors.Keys[0] {
		t.Fatalf("independent=%+v %v", second.Anchors, err)
	}
	encoded, _ := json.Marshal(first)
	var wire map[string]json.RawMessage
	_ = json.Unmarshal(encoded, &wire)
	if _, ok := wire["anchors"]; ok {
		t.Fatal("private receipt response changed")
	}
	changed := submission("first")
	changed.Proposal.Description = "Changed replay"
	if _, _, err := service.Submit(ctx, grant, changed); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed replay=%v", err)
	}
	forged := submission("forged")
	bad := fixture.Derived
	bad.SourceDigest = auditdomain.DigestBytes([]byte("wrong"))
	forged.DerivedAnchors, _ = strictjson.Canonical(bad)
	if _, _, err := service.Submit(ctx, grant, forged); !errors.Is(err, ErrInvalid) {
		t.Fatalf("forged source=%v", err)
	}
	wrongClassification := submission("wrong-classification")
	wrongClassification.Proposal.StandardRefs = nil
	if _, _, err := service.Submit(ctx, grant, wrongClassification); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unrelated issue=%v", err)
	}
	wrongLocation := submission("wrong-location")
	wrongLocation.Proposal.Locations[0].File = "other.py"
	if _, _, err := service.Submit(ctx, grant, wrongLocation); !errors.Is(err, ErrInvalid) {
		t.Fatalf("wrong location=%v", err)
	}
	imitation := submission("imitation")
	imitation.DerivedAnchors = nil
	imitation.EvidenceRefs = refs[1:]
	ordinary, _, err := service.Submit(ctx, grant, imitation)
	if err != nil || ordinary.Anchors.Reason != "not_provided" {
		t.Fatalf("ordinary=%+v %v", ordinary.Anchors, err)
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts").Scan(&count); err != nil || count != 3 {
		t.Fatalf("partial receipt count=%d %v", count, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE finding_proposal_receipts SET anchor_projection = $2 WHERE receipt_id = $1", first.ReceiptID, `{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"not_provided"}`); err == nil {
		t.Fatal("immutable anchor replaced")
	}
	f := deletionImportFixture{ctx: ctx, pool: pool, intake: service, receiptID: first.ReceiptID, request: ImportRequest{OwnerID: owner, RunID: run, Proposal: first.Proposal.Ref}}
	for _, auditID := range []string{"anchor-audit-a", "anchor-audit-b"} {
		createFixtureAudit(t, f, auditID)
		request := f.request
		request.AuditID = auditID
		if _, _, err := service.ImportIntoAudit(ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	var group string
	for _, auditID := range []string{"anchor-audit-a", "anchor-audit-b"} {
		rows, err := querystore.New(pool).ListFindings(ctx, auditstore.FindingListParams{OwnerID: owner, AuditID: auditID, Limit: 10})
		if err != nil || len(rows) != 1 || rows[0].Correlation == nil || rows[0].Correlation.Basis == nil || rows[0].Correlation.Basis.Algorithm != "contractor/snapshot-issue-anchor/v1" || rows[0].Correlation.Basis.Value != fixture.Keys[0].IssueKey || !rows[0].Correlation.HasOtherObservations {
			t.Fatalf("structural primary: %+v %v", rows, err)
		}
		if group != "" && rows[0].Correlation.GroupKey != group {
			t.Fatal("qualified structural observations did not group")
		}
		group = rows[0].Correlation.GroupKey
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
	retained, ok, err := service.Submit(ctx, grant, input)
	if err != nil || !ok || retained.Anchors.Keys[0] != first.Anchors.Keys[0] {
		t.Fatalf("retained replay=%+v %v %v", retained.Anchors, ok, err)
	}
	for _, auditID := range []string{"anchor-audit-a", "anchor-audit-b"} {
		retained, err := service.GetAuditReceipt(ctx, owner, auditID, first.ReceiptID)
		if err != nil || retained.Retention != RetentionAuditHeld || retained.Anchors.Keys[0] != first.Anchors.Keys[0] {
			t.Fatalf("hold=%+v %v", retained.Anchors, err)
		}

		audits := auditstoretest.NewStore(pool).WithPurgeDependencies(DeleteAuditHoldsBatch)
		audit, err := audits.Get(ctx, owner, auditID)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := audits.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: owner, AuditID: auditID, ExpectedRevision: audit.Revision, IdempotencyKey: "purge-" + auditID, RequestDigest: auditdomain.DigestBytes([]byte(auditID))}); err != nil {
			t.Fatal(err)
		}
		claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "anchor-purge", Lease: time.Minute, Limit: 1})
		if err != nil || len(claims) != 1 {
			t.Fatalf("claims=%+v %v", claims, err)
		}
		if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(auditID)); err != nil {
			t.Fatal(err)
		}
	}
	discardedFirst, err := readReceiptByID(ctx, pool, first.ReceiptID)
	if err != nil || discardedFirst.Anchors.Reason != "discarded" || len(discardedFirst.Anchors.Keys) != 0 {
		t.Fatalf("last hold discard=%+v %v", discardedFirst.Anchors, err)
	}
	discarded, err := readReceiptByID(ctx, pool, second.ReceiptID)
	if err != nil || discarded.Anchors.Reason != "discarded" || len(discarded.Anchors.Keys) != 0 {
		t.Fatalf("sensitive discarded=%+v %v", discarded.Anchors, err)
	}
}
