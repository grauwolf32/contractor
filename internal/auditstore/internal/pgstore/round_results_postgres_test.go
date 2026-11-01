package pgstore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

func snapshotStep(f collectingAuditFixture) auditstore.RoundResultsStepParams {
	return auditstore.RoundResultsStepParams{Claim: f.claim, RoundID: *f.audit.CurrentRoundID}
}

func seedPublishedSnapshotForPurge(t *testing.T, f collectingAuditFixture) {
	t.Helper()
	beginSnapshotAssessment(t, f)
	state, err := f.store.AdvanceRoundResults(f.ctx, snapshotStep(f))
	if err != nil || !state.Sealed {
		t.Fatalf("purge snapshot=%+v,%v", state, err)
	}
	publishSnapshotForTest(t, f, state)
}

func TestPostgresRoundResultsPendingCapturePurgesWithAudit(t *testing.T) {
	f := newCollectingAuditFixture(t, "round-results-pending-purge", 8<<20)
	collectPurgeFinding(t, f, "round-results-pending-purge")
	extendSnapshotRound(t, f, 33)
	beginSnapshotAssessment(t, f)
	state, err := f.store.AdvanceRoundResults(f.ctx, snapshotStep(f))
	if err != nil || state.Sealed || state.PageCount != 1 {
		t.Fatalf("pending capture=%+v,%v", state, err)
	}
	claim := drainDeletedAuditForPurge(t, f)
	if err = f.store.PurgeClaimed(f.ctx, claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
		t.Fatal(err)
	}
	assertPurgedTable(t, f, "audit_round_result_pages")
}

func extendSnapshotRound(t *testing.T, f collectingAuditFixture, count int) {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(f.ctx) }()
	cloneWhere(t, f.ctx, tx, "audit_items", "item_id=$1", f.itemID, count-1, map[string]string{
		"item_id": "src.item_id || '-copy-' || n", "item_key": "src.item_key || '-copy-' || n",
		"ordinal": "n-1", "origin": "jsonb_set(src.origin,'{entryKey}',to_jsonb(src.item_key || '-copy-' || n))",
		"state": "'settled'", "final_disposition": "'excluded'", "last_execution_item_id": "NULL",
		"accepted_result_ref": "NULL", "accepted_result_digest": "NULL", "coverage_status": "'excluded'",
	})
	if _, err = tx.Exec(f.ctx, `UPDATE audit_rounds SET expected_item_count=$2 WHERE round_id=$1`, *f.audit.CurrentRoundID, count); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(f.ctx, `UPDATE audits SET max_items_per_round=$2,max_items_total=$2 WHERE audit_id=$1`, f.audit.AuditID, count); err != nil {
		t.Fatal(err)
	}
	if err = tx.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
}

func beginSnapshotAssessment(t *testing.T, f collectingAuditFixture) {
	t.Helper()
	round, err := f.store.GetRound(f.ctx, f.audit.AuditID, *f.audit.CurrentRoundID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.store.TransitionRound(f.ctx, auditstore.RoundTransitionParams{Claim: f.claim, RoundID: round.RoundID, ExpectedRevision: round.Revision, ExpectedState: auditstore.RoundExecuting, TargetState: auditstore.RoundAssessing}); err != nil {
		t.Fatal(err)
	}
}

func publishSnapshotForTest(t *testing.T, f collectingAuditFixture, state auditstore.RoundResultsBuild) auditstore.ExactArtifact {
	t.Helper()
	p := snapshotStep(f)
	service := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	scope, _ := artifacts.ProjectScope(f.audit.ProjectID)
	manifest := auditdomain.RoundResultsManifest{Schema: auditdomain.RoundResultsSchema, AuditID: state.AuditID, RoundID: state.RoundID,
		SourceGeneration: state.Generation, ItemCount: state.ItemCount, MetadataBytes: state.MetadataBytes, Pages: []auditdomain.RoundResultsPageRef{}}
	for ordinal := range state.PageCount {
		raw, err := f.store.ReadRoundResultsPage(f.ctx, p, state.Generation, ordinal)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := auditdomain.DecodeRoundResultsPage(raw)
		if err != nil {
			t.Fatal(err)
		}
		page := auditstore.RoundResultsBuildPage{Ordinal: ordinal, ItemCount: len(decoded.Items)}
		content, err := f.store.ReadRoundResultsPage(f.ctx, p, state.Generation, page.Ordinal)
		if err != nil {
			t.Fatal(err)
		}
		result, err := service.CreateFrozen(f.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: fmt.Sprintf("snapshot-page-%d", page.Ordinal)}, artifacts.Payload{MediaType: "application/json", Data: content})
		if err != nil {
			t.Fatal(err)
		}
		exact := contracts.ExactContent(result.Ref, result.MediaType, content)
		params := auditstore.PublishRoundResultsPageParams{RoundResultsStepParams: p, Generation: state.Generation, Ordinal: page.Ordinal, Artifact: exact}
		if err = f.store.PublishRoundResultsPage(f.ctx, params); err != nil {
			t.Fatal(err)
		}
		// A lost acknowledgement replays the exact natural publication.
		if err = f.store.PublishRoundResultsPage(f.ctx, params); err != nil {
			t.Fatal(err)
		}
		manifest.Pages = append(manifest.Pages, auditdomain.RoundResultsPageRef{Ordinal: page.Ordinal, ItemCount: page.ItemCount, Artifact: exact})
	}
	content, err := auditdomain.EncodeRoundResultsManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.CreateFrozen(f.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "snapshot-root"}, artifacts.Payload{MediaType: "application/json", Data: content})
	if err != nil {
		t.Fatal(err)
	}
	exact := contracts.ExactContent(result.Ref, result.MediaType, content)
	params := auditstore.PublishRoundResultsManifestParams{RoundResultsStepParams: p, Generation: state.Generation, Manifest: exact, Content: manifest}
	for range 2 {
		if err = f.store.PublishRoundResultsManifest(f.ctx, params); err != nil {
			t.Fatal(err)
		}
	}
	return exact
}

func TestPostgresRoundResultsCaptureRestartsBeforeSealAndPinsReviewGeneration(t *testing.T) {
	f := newCollectingAuditFixture(t, "round-results-coherence", 8<<20)
	findingID := collectPurgeFinding(t, f, "round-results-finding")
	extendSnapshotRound(t, f, 33)
	beginSnapshotAssessment(t, f)
	p := snapshotStep(f)
	first, err := f.store.AdvanceRoundResults(f.ctx, p)
	if err != nil || first.Sealed || first.ItemCount != 32 {
		t.Fatalf("first bounded capture=(%+v,%v)", first, err)
	}
	if _, err = f.store.GetAcceptedRoundResults(f.ctx, f.audit.AuditID, p.RoundID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("partial root availability=%v", err)
	}
	if _, err = f.store.ReadRoundResultsPage(f.ctx, p, first.Generation, 0); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("unsealed page availability=%v", err)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE project_findings SET revision=revision+1 WHERE finding_id=$1`, findingID); err != nil {
		t.Fatal(err)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE audits SET revision=revision+1 WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	var state auditstore.RoundResultsBuild
	for range 8 {
		state, err = f.store.AdvanceRoundResults(f.ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		if state.Sealed {
			break
		}
	}
	if !state.Sealed || state.ItemCount != 33 || state.Generation == first.Generation || state.PageCount != 2 {
		t.Fatalf("resumed snapshot=%+v", state)
	}
	content, err := f.store.ReadRoundResultsPage(f.ctx, p, state.Generation, 0)
	if err != nil {
		t.Fatal(err)
	}
	page, err := auditdomain.DecodeRoundResultsPage(content)
	if err != nil {
		t.Fatal(err)
	}
	if page.Items[0].SelectedAttempt == nil || page.Items[0].SelectedAttempt.ReceiptID != "collection-round-results-finding" || len(page.Items[0].Findings) != 1 || page.Items[0].Findings[0].ProjectID != f.project.ProjectID {
		t.Fatalf("selected exact result/review=%+v", page.Items[0])
	}
	before, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	root := publishSnapshotForTest(t, f, state)
	after, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if after.RetainedEvidenceBytes-before.RetainedEvidenceBytes != state.MetadataBytes+root.SizeBytes {
		t.Fatalf("duplicate metadata debit=%d, want %d", after.RetainedEvidenceBytes-before.RetainedEvidenceBytes, state.MetadataBytes+root.SizeBytes)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE project_findings SET revision=revision+1 WHERE finding_id=$1`, findingID); err != nil {
		t.Fatal(err)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE audits SET revision=revision+1 WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	stable, err := f.store.AdvanceRoundResults(f.ctx, p)
	if err != nil || stable.Generation != state.Generation || stable.Manifest == nil || !stable.Manifest.Ref.SameExact(root.Ref) {
		t.Fatalf("accepted snapshot changed=(%+v,%v)", stable, err)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE audit_round_result_pages SET content='corrupt' WHERE round_id=$1 AND ordinal=0`, p.RoundID); err == nil {
		t.Fatal("sealed page content was mutable")
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE audit_rounds SET results_generation=results_generation+1 WHERE round_id=$1`, p.RoundID); err == nil {
		t.Fatal("sealed source generation was mutable")
	}
	unchanged, err := f.store.ReadRoundResultsPage(f.ctx, p, state.Generation, 0)
	if err != nil || !bytes.Equal(content, unchanged) {
		t.Fatalf("later review changed bytes=%v", err)
	}
}

func TestPostgresRoundResultsRejectUnsettledAndOversizeSources(t *testing.T) {
	f := newCollectingAuditFixture(t, "round-results-limits", 8<<20)
	if _, err := f.store.AdvanceRoundResults(f.ctx, snapshotStep(f)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("executing snapshot=%v", err)
	}
	collectPurgeFinding(t, f, "round-results-limits")
	beginSnapshotAssessment(t, f)
	// Stored coverage permits larger arrays than a complete snapshot page;
	// the capture must refuse explicitly without publishing or truncating.
	gaps, _ := json.Marshal([]string{strings.Repeat("x", 600<<10)})
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_items SET coverage_gaps=$2::jsonb WHERE item_id=$1`, f.itemID, gaps); err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.AdvanceRoundResults(f.ctx, snapshotStep(f)); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("oversize source=%v", err)
	}
	if _, err := f.store.GetAcceptedRoundResults(f.ctx, f.audit.AuditID, *f.audit.CurrentRoundID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("oversize root=%v", err)
	}
}

func TestPostgresRoundResultsCaptureAtRoundAndAuditCeilings(t *testing.T) {
	f := newItemScaleFixture(t)
	// Grow the selected Round to 4096 while preserving exactly 100,000 items
	// in its containing Audit; untouched historical Rounds dominate storage.
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(f.ctx) }()
	cloneWhere(t, f.ctx, tx, "audit_items", "item_id=$1", "item-"+f.auditID+"-00000", 971, map[string]string{
		"item_id": "'snapshot-extra-' || n", "item_key": "'snapshot-extra-' || n", "ordinal": "3123+n",
		"origin": "jsonb_set(src.origin,'{entryKey}',to_jsonb('snapshot-extra-' || n))",
		"state":  "'settled'", "final_disposition": "'excluded'", "approval_kind": "'none'", "approval_subject_digest": "NULL",
	})
	if _, err = tx.Exec(f.ctx, `DELETE FROM audit_items WHERE round_id=$1 AND ordinal>=2154`, f.roundIDs[len(f.roundIDs)-1]); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(f.ctx, `UPDATE audit_items SET state='settled',final_disposition='excluded',coverage_rationale=$2 WHERE round_id=$1`, f.roundIDs[0], strings.Repeat("x", 4096)); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(f.ctx, `UPDATE audit_rounds SET state='assessing',expected_item_count=4096 WHERE round_id=$1`, f.roundIDs[0]); err != nil {
		t.Fatal(err)
	}
	if err = tx.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	claims, err := f.store.Claim(f.ctx, auditstore.ClaimParams{HolderID: "snapshot-scale", Lease: 3 * time.Minute, Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	var claim auditstore.ControllerClaim
	for _, candidate := range claims {
		if candidate.AuditID == f.auditID {
			claim = candidate
		}
	}
	if claim.AuditID == "" {
		t.Fatal("scale Audit was not claimed")
	}
	p := auditstore.RoundResultsStepParams{Claim: claim, RoundID: f.roundIDs[0]}
	var state auditstore.RoundResultsBuild
	for range auditdomain.MaximumRoundResultsPages + 1 {
		state, err = f.store.AdvanceRoundResults(f.ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		if state.Sealed {
			break
		}
	}
	if !state.Sealed || state.ItemCount != 4096 || state.PageCount <= 128 || state.PageCount > 256 || state.MetadataBytes > auditdomain.MaximumRoundResultsBytes {
		t.Fatalf("scale snapshot=%+v", state)
	}
	var total int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_items WHERE audit_id=$1`, f.auditID).Scan(&total); err != nil || total != 100_000 {
		t.Fatalf("containing Audit=%d,%v", total, err)
	}
	seen := map[string]bool{}
	for ordinal := range state.PageCount {
		data, err := f.store.ReadRoundResultsPage(f.ctx, p, state.Generation, ordinal)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := auditdomain.DecodeRoundResultsPage(data)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range decoded.Items {
			if seen[item.ItemID] {
				t.Fatalf("duplicate item %s", item.ItemID)
			}
			seen[item.ItemID] = true
		}
	}
	if len(seen) != 4096 {
		t.Fatalf("captured %d selected items", len(seen))
	}
	if _, err = f.pool.Exec(f.ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) audit_items`); err != nil {
		t.Fatal(err)
	}
	for _, mode := range explaintest.Modes {
		analysis := explainPrepared(t, f.ctx, f.pool, mode, roundResultsItemsSQL, f.auditID, p.RoundID, -1)
		if got := analysis.Reads.Of("audit_items"); got > 64+planningEntries {
			t.Fatalf("source rows=%d", got)
		}
		if slices.Contains(analysis.Plan.SeqScans(), "audit_items") {
			t.Fatalf("whole-Audit scan: %s", analysis.Plan)
		}
	}
}
