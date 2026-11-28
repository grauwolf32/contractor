//go:build integration

package auditservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestFindingExportFreezesDuplicateOptionAndCanonicalProjectAssessment(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	const owner, project, audit = "owner", "project", "audit-one"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	createCorrelationAudit(t, ctx, pool, owner, project, "audit-two")
	intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{storage: artifactpgtest.NewStorage(), pool: pool, findings: intake, now: time.Now}
	fingerprint := correlationFingerprint(t)
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "first", auditFindingDocument("first"), fingerprint)
	second := seedAuditFinding(t, ctx, pool, project, owner, audit, "second")
	third := seedAuditFinding(t, ctx, pool, project, owner, audit, "third")
	peer := seedAuditFindingDocument(t, ctx, pool, project, owner, "audit-two", "peer", auditFindingDocument("peer"), fingerprint)
	confirmed := decideFindingForTest(t, ctx, service, owner, audit, first, 1, "confirmed", auditstore.VerdictTruePositive, reviewStringPointer("high"), nil)
	duplicate := decideFindingForTest(t, ctx, service, owner, audit, third, 1, "duplicate", auditstore.VerdictDuplicate, nil, &first)
	if peer != first {
		t.Fatal("qualified source created a second Finding")
	}
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
	if published.SelectedCount != 3 || published.ResultCount != 2 {
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
		if r.FindingID == first && (r.Review.Severity != "high" || r.Review.ActorID == "" || !r.Review.IsState("confirmed")) {
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
	if err := json.Unmarshal(original.Read.Payload.Data, &sarif); err != nil || len(sarif.Runs) != 1 || len(sarif.Runs[0].Results) != 2 {
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
	reassessed := decideFindingForTest(t, ctx, service, owner, "audit-two", peer, confirmed.Revision, "negative", auditstore.VerdictNeedsEvidence, nil, nil)

	replay, err := publisher.PublishFindingExport(ctx, owner, audit, request)
	if err != nil || !replay.Replayed || replay.Artifact.Digest != published.Artifact.Digest {
		t.Fatalf("triage refreshed snapshot: %+v %v", replay, err)
	}
	after, err := publisher.ReadFindingExport(ctx, owner, audit, published.ExportID, "export")
	if err != nil || !bytes.Equal(original.Read.Payload.Data, after.Read.Payload.Data) {
		t.Fatalf("download changed: %v", err)
	}
	peerExport, err := publisher.PublishFindingExport(ctx, owner, "audit-two", findingexport.Request{ClientKey: "frozen", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: peer, Revision: reassessed.Revision}}})
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
	metadata, err := artifactpgtest.NewStorage().Repository(pool).Metadata(ctx, scope, result.Artifact.Ref)
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
	intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{storage: artifactpgtest.NewStorage(), pool: pool, findings: intake, now: time.Now}
	store := auditstoretest.NewStore(pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
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
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
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
	var screenshotPNG bytes.Buffer
	if err := png.Encode(&screenshotPNG, image.NewRGBA(image.Rect(0, 0, 1280, 720))); err != nil {
		t.Fatal(err)
	}
	screenshot := write("ui-screenshot", "image/png", screenshotPNG.Bytes())
	source := write("source", "text/plain", []byte("α😀\r\nsink()"))
	capture := write("capture", auditdomain.HTTPExchangeMediaType, []byte(`{"schema":"contractor.http.exchange.v1","capture":{"source":"http","request_id":"1","request_tag":"tag","observed_at":"2026-11-06T18:20:00Z","attempts":[{"request":{"method":"GET","url":"https://example.test/check?token=private-query","message":{"kind":"structured","headers":[{"name":"Authorization","value":"private-header"}],"body_base64":""}},"response":{"status":200,"message":{"kind":"structured","headers":[{"name":"Content-Type","value":"application/json; private=parameter"}],"body_base64":""},"complete":true}}]},"origin":{"run_id":"run","invocation_id":"invocation"}}`))
	analysis := findinganalysis.Document{Schema: findinganalysis.Schema, ClientKey: "native", Sources: []findinganalysis.Source{{ID: "source", Kind: "file", ArtifactRef: source.Ref}}, Locations: []findinganalysis.Location{{ID: "sink", SourceID: "source", Path: "src/a.py", FileDigest: source.Digest, Region: &findinganalysis.Region{StartLine: 2}}}, Flows: []findinganalysis.Flow{}, HTTPExchanges: []findinganalysis.HTTPExchange{{ID: "exchange", EvidenceRef: capture.Ref, Decoder: "contractor-http-exchange/v1"}}, PrimaryLocationID: "sink", PrimaryHTTPExchangeID: "exchange"}
	encoded, err := strictjson.Canonical(analysis)
	if err != nil {
		t.Fatal(err)
	}
	native := write("native-analysis", findinganalysis.MediaType, encoded)
	seedCollectionRunReceiptDocument(t, ctx, pool, "owner", "source-run", "native", false, true, func(proposal *auditdomain.FindingProposal) {
		proposal.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89"}}
	}, source, capture, native, screenshot)
	intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
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
	// The fixture's raw receipt stands in for explicit Submit. Complete the
	// same owner's admission after its exact Project retention is available.
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		hooks := hookstore.NewFindingEventsTx(tx, artifactpgtest.NewStorage())
		if _, err := hooks.AdmitProjectFinding(ctx, "project", "receipt-native", "finding-native"); err != nil {
			return err
		}
		return hooks.RecordFindingSourceAdded(ctx, "audit", "receipt-native", "source-run")
	}); err != nil {
		t.Fatal(err)
	}

	createCorrelationAudit(t, ctx, pool, "owner", "project", "another-audit")
	if _, _, err := intake.ImportIntoAudit(ctx, findingintake.ImportRequest{OwnerID: "owner", AuditID: "another-audit", RunID: "source-run", Proposal: proposal.Ref}); err != nil {
		t.Fatal(err)
	}
	var peerFinding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM project_findings WHERE project_id='project' AND first_receipt_id='receipt-native'`).Scan(&peerFinding); err != nil {
		t.Fatal(err)
	}
	reviews := &Service{storage: artifactpgtest.NewStorage(), pool: pool, findings: intake, now: time.Now}
	decideFindingForTest(t, ctx, reviews, "owner", "another-audit", peerFinding, 1, "other-native", auditstore.VerdictFalsePositive, nil, nil)
	var finding string
	if err := pool.QueryRow(ctx, `SELECT finding_id FROM project_findings WHERE project_id='project' AND first_receipt_id='receipt-native'`).Scan(&finding); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).TransitionRun(ctx, "source-run", runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "test_done"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "owner", "source-run"); err != nil {
		t.Fatal(err)
	}
	publisher := newCollectionTestPublisher(t, pool)
	result, err := publisher.PublishFindingExport(ctx, "owner", "audit", findingexport.Request{ClientKey: "native", Format: "sarif", Profile: "generic", Findings: []findingexport.Selection{{FindingID: finding, Revision: 2}}})
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
	if !snapshot.Results[0].Review.IsState("rejected") || snapshot.Results[0].Review.ActorID != "owner" {
		t.Fatal("export did not read the same canonical Project assessment")
	}
	r := snapshot.Receipts[0]
	if r.Analysis == nil || r.Analysis.Analysis.PrimaryLocationID != "sink" || r.Analysis.Sources[0].Document.Scope.Kind != "run" || r.HTTP == nil || r.HTTP.URI != "https://example.test/check" || r.HTTP.ContentType != "application/json" {
		t.Fatalf("lost original/safe projection: %+v", r)
	}
	if bytes.Contains(downloaded.Read.Payload.Data, []byte("private-")) || bytes.Contains(downloaded.Read.Payload.Data, []byte("private=parameter")) {
		t.Fatal("snapshot copied private HTTP data")
	}
	content, err := publisher.ReadFindingExport(ctx, "owner", "audit", result.ExportID, "export")
	if err != nil {
		t.Fatal(err)
	}
	var sarif struct {
		Runs []struct {
			Tool struct {
				Driver map[string]json.RawMessage `json:"driver"`
			} `json:"tool"`
			Taxonomies []json.RawMessage `json:"taxonomies"`
			Results    []struct {
				Taxa []json.RawMessage `json:"taxa"`
			} `json:"results"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(content.Read.Payload.Data, &sarif); err != nil {
		t.Fatal(err)
	}
	if len(sarif.Runs) != 1 || len(sarif.Runs[0].Taxonomies) != 1 || len(sarif.Runs[0].Results) != 1 || len(sarif.Runs[0].Results[0].Taxa) != 1 {
		t.Fatal("lost exact result classification")
	}
	if _, exists := sarif.Runs[0].Tool.Driver["supportedTaxonomies"]; exists {
		t.Fatal("result classification became a shared rule claim")
	}
	retainedScreenshot := false
	project, err := service.Project("project")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range r.Evidence {
		if e.Original.Ref.Name == "ui-screenshot" {
			read, err := project.Read(ctx, e.Retained.Ref)
			if err != nil || read.Payload.MediaType != "image/png" || !bytes.Equal(read.Payload.Data, screenshotPNG.Bytes()) {
				t.Fatalf("screenshot lost exact PNG bytes after source Run deletion: %v", err)
			}
			retainedScreenshot = true
		}
		if e.Original.Scope.Kind != "run" || e.Retained.Scope.Kind != "project" || e.Retained.Ref.Namespace != artifactpolicy.FindingMaterialsNamespace {
			t.Fatal("exact original/retained evidence authority lost")
		}
	}
	if !retainedScreenshot {
		t.Fatal("screenshot absent from retained Finding evidence")
	}
}
