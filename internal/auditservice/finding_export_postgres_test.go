//go:build integration

package auditservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestFindingExportFreezesDuplicateOptionAndIndependentAuditReviews(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	const owner, project, audit = "owner", "project", "audit-one"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	createCorrelationAudit(t, ctx, pool, owner, project, "audit-two")
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	fingerprint := correlationFingerprint(t)
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "first", auditFindingDocument("first"), fingerprint)
	second := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "second", auditFindingDocument("second"), fingerprint)
	third := seedAuditFinding(t, ctx, pool, project, owner, audit, "third")
	peer := seedAuditFindingDocument(t, ctx, pool, project, owner, "audit-two", "peer", auditFindingDocument("peer"), fingerprint)
	confirmed := decideFindingForTest(t, ctx, service, owner, audit, first, 1, "confirmed", auditstore.VerdictTruePositive, reviewStringPointer("high"), nil)
	duplicate := decideFindingForTest(t, ctx, service, owner, audit, third, 1, "duplicate", auditstore.VerdictDuplicate, nil, &first)
	rejected := decideFindingForTest(t, ctx, service, owner, "audit-two", peer, 1, "rejected", auditstore.VerdictFalsePositive, nil, nil)
	publisher := newCollectionTestPublisher(t, pool)
	request := findingexport.Request{ClientKey: "frozen", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: first, Revision: confirmed.Revision}, {FindingID: second, Revision: 1}, {FindingID: third, Revision: duplicate.Revision}}}
	var results [2]findingintake.PublishedExport
	var failures [2]error
	var wg sync.WaitGroup
	for i := range 2 {
		wg.Go(func() { results[i], failures[i] = publisher.PublishFindingExport(ctx, owner, audit, request) })
	}
	wg.Wait()
	if failures[0] != nil || failures[1] != nil || results[0].Replayed == results[1].Replayed || results[0].Artifact.Digest != results[1].Artifact.Digest {
		t.Fatalf("atomic replay: %+v %v", results, failures)
	}
	published := results[0]
	if published.SelectedCount != 3 || published.ResultCount != 1 {
		t.Fatalf("duplicate count: %+v", published)
	}
	download, err := publisher.ReadFindingExport(ctx, owner, audit, published.ExportID, "export-snapshot")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := findingexport.DecodeSnapshot(download.Read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Results) != 3 || snapshot.Request.IncludeDuplicates {
		t.Fatal("snapshot discarded selection or option")
	}
	for _, r := range snapshot.Results {
		if r.FindingID == first && (r.Review.Severity != "high" || r.Review.DecisionID == "" || r.Review.State != "confirmed") {
			t.Fatalf("lost own review: %+v", r)
		}
	}
	original, err := publisher.ReadFindingExport(ctx, owner, audit, published.ExportID, "export")
	if err != nil {
		t.Fatal(err)
	}
	var sarif struct {
		Runs []struct {
			Results []json.RawMessage `json:"results"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(original.Read.Payload.Data, &sarif); err != nil || len(sarif.Runs) != 1 || len(sarif.Runs[0].Results) != 1 {
		t.Fatalf("rendered count: %+v %v", sarif, err)
	}
	changed := request
	changed.IncludeDuplicates = true
	if _, err := publisher.PublishFindingExport(ctx, owner, audit, changed); !errors.Is(err, findingintake.ErrCollectionKeyConflict) {
		t.Fatalf("changed option: %v", err)
	}
	changed.ClientKey = "expanded"
	expanded, err := publisher.PublishFindingExport(ctx, owner, audit, changed)
	if err != nil || expanded.ResultCount != 3 {
		t.Fatalf("include duplicate: %+v %v", expanded, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_findings SET state='needs-evidence',revision=revision+1,current_decision_id=NULL WHERE finding_id=$1`, first); err != nil {
		t.Fatal(err)
	}
	replay, err := publisher.PublishFindingExport(ctx, owner, audit, request)
	if err != nil || !replay.Replayed || replay.Artifact.Digest != published.Artifact.Digest {
		t.Fatalf("triage refreshed snapshot: %+v %v", replay, err)
	}
	after, err := publisher.ReadFindingExport(ctx, owner, audit, published.ExportID, "export")
	if err != nil || !bytes.Equal(original.Read.Payload.Data, after.Read.Payload.Data) {
		t.Fatalf("download changed: %v", err)
	}
	peerExport, err := publisher.PublishFindingExport(ctx, owner, "audit-two", findingexport.Request{ClientKey: "frozen", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: peer, Revision: rejected.Revision}}})
	if err != nil || peerExport.ResultCount != 1 || peerExport.ExportID == published.ExportID {
		t.Fatalf("independent Audit export: %+v %v", peerExport, err)
	}
	for _, tc := range []struct{ owner, audit string }{{"foreign", audit}, {owner, "audit-two"}} {
		if _, err := publisher.ReadFindingExport(ctx, tc.owner, tc.audit, published.ExportID, "export"); err == nil {
			t.Fatal("foreign export disclosed")
		}
	}
}

func TestFindingExportInvalidInputsAndAtomicRollback(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	createCorrelationAudit(t, ctx, pool, "owner", "project", "audit")
	id := seedAuditFinding(t, ctx, pool, "project", "owner", "audit", "first")
	publisher := newCollectionTestPublisher(t, pool)
	request := findingexport.Request{ClientKey: "rollback", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: id, Revision: 1}}}
	for _, tc := range []struct {
		name      string
		selection findingexport.Selection
		want      error
	}{{"missing", findingexport.Selection{FindingID: "unknown", Revision: 1}, findingintake.ErrNotFound}, {"stale", findingexport.Selection{FindingID: id, Revision: 2}, findingintake.ErrConflict}, {"presenter", findingexport.Selection{FindingID: id, Revision: 1, PresentationReceiptID: "not-a-member"}, findingintake.ErrNotFound}} {
		r := request
		r.ClientKey = tc.name
		r.Findings = []findingexport.Selection{tc.selection}
		if _, err := publisher.PublishFindingExport(ctx, "owner", "audit", r); !errors.Is(err, tc.want) {
			t.Fatalf("%s: %v", tc.name, err)
		}
	}
	if _, err := pool.Exec(ctx, `CREATE FUNCTION fail_export_receipt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.name LIKE 'finding-export-receipt.%' THEN RAISE EXCEPTION 'injected failure'; END IF; RETURN NEW; END $$; CREATE TRIGGER fail_export_receipt BEFORE INSERT ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION fail_export_receipt()`); err != nil {
		t.Fatal(err)
	}
	if _, err := publisher.PublishFindingExport(ctx, "owner", "audit", request); err == nil {
		t.Fatal("injected failure succeeded")
	}
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE namespace=$1`, auditdomain.ArtifactNamespace("audit")).Scan(&count); err != nil || count != 0 {
		t.Fatalf("partial publication: %d %v", count, err)
	}
	if _, err := pool.Exec(ctx, `DROP TRIGGER fail_export_receipt ON artifact_bindings`); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := publisher.PublishFindingExport(cancelled, "owner", "audit", request); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	result, err := publisher.PublishFindingExport(ctx, "owner", "audit", request)
	if err != nil || result.Replayed {
		t.Fatalf("retry rollback: %+v %v", result, err)
	}
	for _, part := range []string{"export", "export-snapshot", "export-diagnostics"} {
		value, err := publisher.ReadFindingExport(ctx, "owner", "audit", result.ExportID, part)
		if err != nil || len(value.Read.Payload.Data) == 0 {
			t.Fatalf("download %s: %v", part, err)
		}
	}
	scope, _ := artifacts.ProjectScope("project")
	metadata, err := artifacts.NewPostgresRepository(pool).Metadata(ctx, scope, result.Artifact.Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("not frozen: %+v %v", metadata, err)
	}
}

func TestFindingExportMaximumSelectionAndAuditPurge(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	createCorrelationAudit(t, ctx, pool, "owner", "project", "audit")
	request := findingexport.Request{ClientKey: "maximum", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{}}
	for i := range auditdomain.MaximumCollectionEntries {
		id := seedAuditFinding(t, ctx, pool, "project", "owner", "audit", fmt.Sprintf("maximum-%03d", i))
		request.Findings = append(request.Findings, findingexport.Selection{FindingID: id, Revision: 1})
	}
	publisher := newCollectionTestPublisher(t, pool)
	result, err := publisher.PublishFindingExport(ctx, "owner", "audit", request)
	if err != nil || result.SelectedCount != 256 || result.ResultCount != 256 {
		t.Fatalf("maximum selection: %+v %v", result, err)
	}
	tooMany := request
	tooMany.ClientKey = "over-limit"
	tooMany.Findings = append(append([]findingexport.Selection{}, request.Findings...), findingexport.Selection{FindingID: "extra", Revision: 1})
	if _, err := publisher.PublishFindingExport(ctx, "owner", "audit", tooMany); err == nil {
		t.Fatal("accepted 257 selectors")
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	store := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	audit, err := store.Get(ctx, "owner", "audit")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Delete(ctx, MutationParams{OwnerID: "owner", AuditID: "audit", ExpectedRevision: audit.Revision, IdempotencyKey: "delete", RequestDigest: serviceTestDigest("delete")}); err != nil {
		t.Fatal(err)
	}
	if _, err := publisher.ReadFindingExport(ctx, "owner", "audit", result.ExportID, "export"); err == nil {
		t.Fatal("deleting Audit export remained readable")
	}
	claim := claimAuditForPurge(t, ctx, store, "audit")
	for tick := 0; tick < 16; tick++ {
		if err := store.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace("audit")); err != nil {
			t.Fatal(err)
		}
		_, err := store.Get(ctx, "owner", "audit")
		if errors.Is(err, auditstore.ErrNotFound) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if tick == 15 {
			t.Fatal("bounded Audit purge did not finish")
		}
	}
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE namespace=$1`, auditdomain.ArtifactNamespace("audit")).Scan(&count); err != nil || count != 0 {
		t.Fatalf("export survived purge: %d %v", count, err)
	}
}

func TestFindingExportRetainsResolvedCodeAndSafeHTTPAfterSourceDeletion(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	createCorrelationAudit(t, ctx, pool, "owner", "project", "audit")
	seedCollectionRun(t, ctx, pool, "owner", "source-run", "project")
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	run, err := service.Run("source-run")
	if err != nil {
		t.Fatal(err)
	}
	write := func(name, media string, data []byte) findingintake.ExactArtifact {
		t.Helper()
		result, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: name}, artifacts.Payload{MediaType: media, Data: data}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return findingintake.ExactArtifact{Ref: result.Ref, Digest: auditdomain.DigestBytes(data), MediaType: media, SizeBytes: int64(len(data))}
	}
	source := write("source", "text/plain", []byte("α😀\r\nsink()"))
	capture := write("capture", auditdomain.HTTPExchangeMediaType, []byte(`{"schema":"contractor.http.exchange.v1","capture":{"source":"http","request_id":"1","request_tag":"tag","attempts":[{"request":{"method":"GET","url":"https://example.test/check?token=private-query","message":{"kind":"structured","headers":[{"name":"Authorization","value":"private-header"}],"body_base64":""}},"response":{"status":200,"message":{"kind":"structured","headers":[{"name":"Content-Type","value":"application/json; private=parameter"}],"body_base64":""},"complete":true}}]},"finding":{"proposal_id":"original-proposal","receipt_id":"original-receipt"}}`))
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, ClientKey: "native", Sources: []findinganalysis.Source{{ID: "source", Kind: "file", ArtifactRef: source.Ref}}, Locations: []findinganalysis.Location{{ID: "sink", SourceID: "source", Path: "src/a.py", FileDigest: source.Digest, Region: &findinganalysis.Region{StartLine: 2}}}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{{ID: "exchange", EvidenceRef: capture.Ref, Decoder: "contractor-http-exchange/v1"}}, PrimaryLocationID: "sink", PrimaryHTTPExchangeID: "exchange"}
	encoded, err := strictjson.Canonical(analysis)
	if err != nil {
		t.Fatal(err)
	}
	native := write("native-analysis", findinganalysis.MediaType, encoded)
	seedCollectionRunReceipt(t, ctx, pool, "owner", "source-run", "native", false, source, capture, native)
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "finding-proposals", Name: "native"})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := intake.ImportIntoAudit(ctx, findingintake.ImportRequest{OwnerID: "owner", AuditID: "audit", RunID: "source-run", Proposal: proposal.Ref}); err != nil {
		t.Fatal(err)
	}
	createCorrelationAudit(t, ctx, pool, "owner", "project", "another-audit")
	if _, _, err := intake.ImportIntoAudit(ctx, findingintake.ImportRequest{OwnerID: "owner", AuditID: "another-audit", RunID: "source-run", Proposal: proposal.Ref}); err != nil {
		t.Fatal(err)
	}
	var peerFinding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM audit_findings WHERE audit_id='another-audit' AND first_receipt_id='receipt-native'`).Scan(&peerFinding); err != nil {
		t.Fatal(err)
	}
	reviews := &Service{pool: pool, findings: intake, now: time.Now}
	decideFindingForTest(t, ctx, reviews, "owner", "another-audit", peerFinding, 1, "other-native", auditstore.VerdictFalsePositive, nil, nil)
	var finding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM audit_findings WHERE audit_id='audit' AND first_receipt_id='receipt-native'`).Scan(&finding); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool).TransitionRun(ctx, "source-run", runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "test_done"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(ctx, "owner", "source-run"); err != nil {
		t.Fatal(err)
	}
	publisher := newCollectionTestPublisher(t, pool)
	result, err := publisher.PublishFindingExport(ctx, "owner", "audit", findingexport.Request{ClientKey: "native", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: finding, Revision: 1}}})
	if err != nil {
		t.Fatal(err)
	}
	downloaded, err := publisher.ReadFindingExport(ctx, "owner", "audit", result.ExportID, "export-snapshot")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := findingexport.DecodeSnapshot(downloaded.Read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Results[0].Review.State != "proposed" || snapshot.Results[0].Review.DecisionID != "" {
		t.Fatal("borrowed another Audits decision for the same receipt")
	}
	r := snapshot.Receipts[0]
	if r.Analysis == nil || r.Analysis.Analysis.PrimaryLocationID != "sink" || r.Analysis.Sources[0].Document.Scope.Kind != "run" || r.HTTP == nil || r.HTTP.URI != "https://example.test/check" || r.HTTP.ContentType != "application/json" {
		t.Fatalf("lost original/safe projection: %+v", r)
	}
	if bytes.Contains(downloaded.Read.Payload.Data, []byte("private-")) || bytes.Contains(downloaded.Read.Payload.Data, []byte("private=parameter")) {
		t.Fatal("snapshot copied private HTTP data")
	}
	for _, e := range r.Evidence {
		if e.Original.Scope.Kind != "run" || e.Retained.Scope.Kind != "project" || e.Retained.Ref.Namespace != auditdomain.ArtifactNamespace("audit") {
			t.Fatal("exact original/retained evidence authority lost")
		}
	}
}
