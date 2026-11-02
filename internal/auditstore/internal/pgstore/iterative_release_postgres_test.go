//go:build integration

package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// The input population is deliberately much larger than a changing slice. The
// first 128 records are already rejected: filtering them must not authorize an
// unbounded search for the next admissible record or any proposal payload read.
func TestPostgresIterativeAdmissionBoundsCandidatesAtContainingCeiling(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	f, admission, params := admissionFixture(t, "iterative-candidate-scale")
	f.ctx = ctx
	if err := f.store.RenewClaim(ctx, f.claim, 3*time.Minute); err != nil {
		t.Fatal(err)
	}
	const population = 100_000
	const firstID = "receipt-scale-000001"
	stagedProposal(t, f, 1, firstID)
	tx, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	cloneWhere(t, ctx, tx, "finding_proposal_receipts", "receipt_id=$1", firstID, population-1, map[string]string{
		"receipt_id": "'receipt-scale-'||lpad(n::text,6,'0')", "proposal_id": "'proposal-scale-'||n",
		"allocation_id": "'allocation-scale-'||n", "invocation_id": "'invocation-scale-'||n", "submission_id": "'submission-scale-'||n",
	})
	// These are ordinary retained proposals, with no publication intent.
	// Holds alone must never materialize Finding identities or events.
	cloneWhere(t, ctx, tx, "finding_proposal_audit_holds", "receipt_id=$1", firstID, population-1, map[string]string{"receipt_id": "'receipt-scale-'||lpad(n::text,6,'0')"})
	if err = tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	var holds, receipts, findings int
	if err = f.pool.QueryRow(ctx, `SELECT
      (SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id=$1),
      (SELECT count(*) FROM finding_proposal_receipts WHERE project_id=$2),
      (SELECT count(*) FROM project_findings WHERE project_id=$2)`, f.audit.AuditID, f.audit.ProjectID).Scan(&holds, &receipts, &findings); err != nil || holds != population || receipts != population || findings != 0 {
		t.Fatalf("ordinary proposal population holds=%d receipts=%d findings=%d error=%v", holds, receipts, findings, err)
	}
	// Unrelated retained links make the decision status lookup use its natural
	// provenance index rather than relying on the complete link table being tiny.
	if _, err = f.pool.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
 SELECT hold.audit_id,'unrelated/'||n,hold.proposal_ref->'ref',hold.proposal_ref->>'digest',hold.proposal_ref->>'mediaType',(hold.proposal_ref->>'sizeBytes')::bigint,'{}'
 FROM finding_proposal_audit_holds AS hold CROSS JOIN generate_series(1,2000) AS n WHERE hold.receipt_id=$1`, firstID); err != nil {
		t.Fatal(err)
	}
	state, err := admission.WithStep(ctx, params, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		t.Fatal("first intent must precede callback")
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var statusInput []map[string]any
	state, err = admission.WithStep(ctx, params, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		page, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 128)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if len(page.Candidates) != 128 || page.Exhausted {
			t.Fatalf("first raw prefix=%d exhausted=%v", len(page.Candidates), page.Exhausted)
		}
		decisions := make([]auditstore.AdmissionDecision, len(page.Candidates))
		for i, c := range page.Candidates {
			if c.Ordinal != 0 || c.AlreadyRejected || c.AlreadyConsumed {
				t.Fatalf("unexpected fresh source=%+v", c)
			}
			decisions[i] = auditstore.AdmissionDecision{ReceiptID: c.ReceiptID, ProposedCheckOrdinal: 0, Proposal: c.Proposal, SourceCheckDigest: c.Header.Fragment.Digest, PlanDigest: c.Header.PlanDigest, Disposition: "rejected", Code: "candidate_metadata_limit"}
			statusInput = append(statusInput, map[string]any{"position": i, "receipt_id": c.ReceiptID, "ordinal": 0, "source_key": admissionSourceKey(c.ReceiptID, 0, c.Proposal.Digest, c.Header.Fragment.Digest)})
		}
		if len(s.reader.payloads) != 0 || s.reader.work.primary != 128 {
			t.Fatalf("raw lookup read payloads or exceeded primary work: payloads=%d work=%+v", len(s.reader.payloads), s.reader.work)
		}
		return auditstore.AdmissionMutation{Decisions: decisions}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	originalCursor := state.Cursor
	state, err = admission.WithStep(ctx, params, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		page, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 128)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if len(page.Candidates) != 128 || page.Exhausted {
			t.Fatalf("rejected raw prefix=%d exhausted=%v", len(page.Candidates), page.Exhausted)
		}
		for _, c := range page.Candidates {
			if !c.AlreadyRejected {
				t.Fatalf("rejected prefix skipped before bound: %+v", c)
			}
		}
		if len(s.reader.payloads) != 0 || s.reader.work.primary != 128 || s.reader.work.related > 512 || s.reader.work.metadata > 1<<20 {
			t.Fatalf("bounded raw work=%+v payloads=%d", s.reader.work, len(s.reader.payloads))
		}
		next := s.State
		next.Cursor = page.Candidates[127].Cursor
		return auditstore.AdmissionMutation{Next: next}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = admission.WithStep(ctx, params, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		page, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 128)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if len(page.Candidates) != 128 || page.Candidates[0].ReceiptID != "receipt-scale-000129" || page.Candidates[0].AlreadyRejected {
			t.Fatalf("bounded continuation lost next source: %+v", page.Candidates[0])
		}
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"finding_proposal_audit_holds", "finding_proposal_receipts", "audit_artifact_links", "audit_items"} {
		if _, err = f.pool.Exec(ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) `+table); err != nil {
			t.Fatal(err)
		}
	}
	statusJSON, _ := json.Marshal(statusInput)
	for _, mode := range explaintest.Modes {
		headers := explainPrepared(t, ctx, f.pool, mode, admissionCandidateHeadersSQL, state.AuditID, f.audit.ProjectID, originalCursor.CreatedAt, originalCursor.ReceiptID, state.SourceUpper.CreatedAt, state.SourceUpper.ReceiptID, state.SourceCutoffAt, 128, originalCursor.Ordinal)
		for _, table := range []string{"finding_proposal_audit_holds", "finding_proposal_receipts"} {
			if reads := headers.Reads.Of(table); reads > 128+planningEntries {
				t.Fatalf("%s raw %s rows=%d", mode, table, reads)
			}
			if slices.Contains(headers.Plan.SeqScans(), table) {
				t.Fatalf("%s whole candidate population scan: %s", mode, headers.Plan)
			}
		}
		if reads := headers.Reads.Of("artifact_blobs"); reads != 0 {
			t.Fatalf("%s header query read %d payload rows", mode, reads)
		}
		statuses := explainPrepared(t, ctx, f.pool, mode, admissionSourceStatusesSQL, state.AuditID, string(statusJSON), admissionPolicyDigest(state))
		if reads := statuses.Reads.Of("artifact_blobs"); reads != 0 {
			t.Fatalf("%s status query read %d payload rows", mode, reads)
		}
		if reads := statuses.Reads.Of("audit_artifact_links"); reads > 128+planningEntries {
			t.Fatalf("%s decision index rows=%d", mode, reads)
		}
		t.Logf("%s reads: holds=%d receipts=%d decision-links=%d payloads=%d", mode, headers.Reads.Of("finding_proposal_audit_holds"), headers.Reads.Of("finding_proposal_receipts"), statuses.Reads.Of("audit_artifact_links"), headers.Reads.Of("artifact_blobs")+statuses.Reads.Of("artifact_blobs"))
		if slices.Contains(statuses.Plan.SeqScans(), "audit_artifact_links") {
			t.Fatalf("%s whole decision population scan: %s", mode, statuses.Plan)
		}
	}
}

// The first page's transaction commits but its acknowledgement is lost. Only
// the replacement epoch can resume, and replay cannot debit that exact page
// again or alter the complete immutable generation.
func TestPostgresIterativeSnapshotPublicationReplaysAfterClaimTakeover(t *testing.T) {
	f := newCollectingAuditFixture(t, "iterative-snapshot-claim", 16<<20)
	collectPurgeFinding(t, f, "iterative-snapshot-source")
	extendSnapshotRound(t, f, 33)
	beginSnapshotAssessment(t, f)
	p := snapshotStep(f)
	var source auditstore.RoundResultsBuild
	for range 3 {
		var err error
		source, err = f.store.AdvanceRoundResults(f.ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		if source.Sealed {
			break
		}
	}
	if !source.Sealed || source.PageCount != 2 || source.ItemCount != 33 {
		t.Fatalf("two-page source=%+v", source)
	}
	before, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	data, err := f.store.ReadRoundResultsPage(f.ctx, p, source.Generation, 0)
	if err != nil {
		t.Fatal(err)
	}
	artifactsService := artifacts.NewService(artifactpostgres.NewRepository(f.pool))
	scope, _ := artifacts.ProjectScope(f.audit.ProjectID)
	written, err := artifactsService.CreateFrozen(f.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "snapshot-page-0"}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		t.Fatal(err)
	}
	page := auditstore.PublishRoundResultsPageParams{RoundResultsStepParams: p, Generation: source.Generation, Ordinal: 0, Artifact: contracts.ExactContent(written.Ref, written.MediaType, data)}
	lost := errors.New("page publication acknowledgement lost")
	publishWithLostReply := func() error {
		if err := f.store.PublishRoundResultsPage(f.ctx, page); err != nil {
			return err
		}
		return lost
	}
	if err = publishWithLostReply(); !errors.Is(err, lost) {
		t.Fatal(err)
	}
	stale := f.claim
	if _, err = f.pool.Exec(f.ctx, `UPDATE audit_controller_claims SET claimed_at=clock_timestamp()-interval '2 seconds',expires_at=clock_timestamp()-interval '1 second' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: "snapshot-replacement", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 || claims[0].Epoch <= stale.Epoch {
		t.Fatalf("claim takeover=%+v error=%v", claims, err)
	}
	f.claim = claims[0]
	f.store = newTestStore(f.pool)
	if err = f.store.PublishRoundResultsPage(f.ctx, page); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale publication=%v", err)
	}
	if _, err = f.store.AdvanceRoundResults(f.ctx, p); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale capture=%v", err)
	}
	page.Claim = f.claim
	if err = f.store.PublishRoundResultsPage(f.ctx, page); err != nil {
		t.Fatal(err)
	}
	recovered, err := f.store.AdvanceRoundResults(f.ctx, snapshotStep(f))
	if err != nil || recovered.Generation != source.Generation || len(recovered.Pages) != 2 || recovered.Pages[1].Artifact != nil || recovered.Pages[0].Artifact == nil || recovered.Pages[0].Artifact.Verify(page.Artifact) != nil {
		t.Fatalf("lost page reply recovery=%+v error=%v", recovered, err)
	}
	if _, err = f.store.GetAcceptedRoundResults(f.ctx, f.audit.AuditID, p.RoundID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("partial publication exposed root: %v", err)
	}
	root := publishSnapshotForTest(t, f, recovered)
	accepted, err := f.store.GetAcceptedRoundResults(f.ctx, f.audit.AuditID, p.RoundID)
	if err != nil || accepted.Verify(root) != nil {
		t.Fatalf("accepted exact root=%+v error=%v", accepted, err)
	}
	after, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if debit := after.RetainedEvidenceBytes - before.RetainedEvidenceBytes; debit != source.MetadataBytes+root.SizeBytes {
		t.Fatalf("takeover/replay debit=%d want=%d", debit, source.MetadataBytes+root.SizeBytes)
	}
	var links int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id=$1 AND logical_key LIKE $2`, f.audit.AuditID, "round-results/"+p.RoundID+"%").Scan(&links); err != nil || links != 3 {
		t.Fatalf("accepted publication links=%d error=%v", links, err)
	}
}
