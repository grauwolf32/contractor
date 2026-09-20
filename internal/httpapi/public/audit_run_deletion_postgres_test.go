//go:build integration

package public

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Commit real deletion between the provenance reader's revision snapshot and
// receipt hydration, using a separate pool and no sleeps or production hooks.
type publicRunDeletionTrace struct {
	delete func(context.Context) error
	called bool
	err    error
}

func (trace *publicRunDeletionTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if trace.delete != nil && strings.Contains(data.SQL, "WITH anchors AS") {
		remove := trace.delete
		trace.delete = nil
		trace.called = true
		trace.err = remove(ctx)
	}
	return ctx
}

func (*publicRunDeletionTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestPublicAuditSourceRunDeletionInvalidatesReadContexts(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, projectID, runID = "user-1", "deletion-project", "deletion-source"
	store := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	for _, pair := range [][2]string{{owner, projectID}, {"other-owner", "other-project"}} {
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
			OwnerID: pair[0], ProjectID: pair[1], Kind: projectstore.KindProject, Name: pair[1],
			IdempotencyKey: pair[1], RequestDigest: auditHandlerDigest(pair[1]),
		}); err != nil {
			t.Fatal(err)
		}
	}
	projectArtifacts, err := store.Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	input, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	selection, err := json.Marshal(auditservice.DraftSelection{
		Schema:        auditservice.DraftSelectionSchema,
		Inputs:        map[string]auditstore.ExactArtifact{"source": {Ref: input.Ref, Digest: auditHandlerDigest("source"), MediaType: input.MediaType, SizeBytes: input.Size}},
		RuntimeLabels: []string{},
	})
	if err != nil {
		t.Fatal(err)
	}
	auditOwners := map[string]string{"audit-a": owner, "audit-b": owner, "unaffected": owner, "foreign-audit": "other-owner"}
	for auditID, auditOwner := range auditOwners {
		destination := projectID
		if auditOwner != owner {
			destination = "other-project"
		}
		if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: auditOwner, ProjectID: destination,
			Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditHandlerDigest("profile")},
			ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: selection,
			Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey: auditID, RequestDigest: auditHandlerDigest(auditID),
		}); err != nil {
			t.Fatal(err)
		}
	}
	trace := &publicRunDeletionTrace{}
	configuration := pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	readerPool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	defer readerPool.Close()
	var audits *auditservice.Service
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(deps *Dependencies) {
		credentials := newFakeManagedCredentials()
		audits, err = auditservice.New(auditservice.Options{
			Pool: readerPool, Profiles: deps.Config.(*config.Manager), CredentialGuard: credentials,
			TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil }),
		})
		if err != nil {
			t.Fatal(err)
		}
		deps.Audits = audits
	})
	snapshot, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	project := projectID
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: owner, ProjectID: &project, WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: workflowJSON, Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	destinations := map[string]string{"source-a": "audit-a", "source-extra": "audit-a", "source-b": "audit-b"}
	beforeReceipts := make(map[string]findingintake.Receipt)
	for suffix, auditID := range destinations {
		ref := seedPublicDeletionRunReceipt(t, ctx, pool, owner, runID, suffix)
		if _, replayed, err := intake.ImportIntoAudit(ctx, findingintake.ImportRequest{OwnerID: owner, AuditID: auditID, RunID: runID, Proposal: ref}); err != nil || replayed {
			t.Fatalf("import %s: replayed=%t err=%v", suffix, replayed, err)
		}
		receipt, err := intake.GetAuditReceipt(ctx, owner, auditID, "receipt-"+suffix)
		if err != nil || receipt.Origin.RunDeleted {
			t.Fatalf("initial receipt %s: %+v err=%v", suffix, receipt, err)
		}
		beforeReceipts[suffix] = receipt
	}
	// Use the existing contributing-receipt relation to get a genuine signed
	// continuation for one finding, retaining both actually imported holds.
	const findingID = "finding-receipt-source-a"
	if _, err := pool.Exec(ctx, `DELETE FROM audit_findings WHERE finding_id='finding-receipt-source-extra'`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_finding_contributions(finding_id,audit_id,receipt_id,relation,proposal_ref)
SELECT $1,'audit-a',receipt_id,'contributing',proposal_ref FROM finding_proposal_audit_holds
WHERE receipt_id='receipt-source-extra' AND audit_id='audit-a'`, findingID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "source-done"}); err != nil {
		t.Fatal(err)
	}
	get := func(path string) *httptest.ResponseRecorder {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, path, nil).WithContext(ctx))
		return response
	}
	before := make(map[string]auditstore.Audit)
	etags := make(map[string]string)
	for auditID, auditOwner := range auditOwners {
		audit, err := audits.Get(ctx, auditOwner, auditID)
		if err != nil {
			t.Fatal(err)
		}
		before[auditID] = audit
		if auditOwner == owner {
			response := get("/v1/audits/" + auditID)
			if response.Code != http.StatusOK || response.Header().Get("ETag") != strconv.Quote(strconv.FormatUint(audit.Revision, 10)) {
				t.Fatalf("initial Audit %s: %d %s", auditID, response.Code, response.Body.String())
			}
			etags[auditID] = response.Header().Get("ETag")
		}
	}
	path := "/v1/audits/audit-a/findings/" + findingID + "/provenance"
	initialResponse := get(path + "?limit=1")
	var initial findingProvenancePageResponse
	if err := json.Unmarshal(initialResponse.Body.Bytes(), &initial); err != nil || initialResponse.Code != http.StatusOK || len(initial.Items) != 1 || initial.Page.NextCursor == nil {
		t.Fatalf("initial cursor: %d %s err=%v", initialResponse.Code, initialResponse.Body.String(), err)
	}
	pinnedPath := fmt.Sprintf("%s?auditRevision=%d&findingRevision=%d", path, initial.AuditRevision, initial.FindingRevision)
	trace.delete = func(ctx context.Context) error { return runs.DeleteReleasedTerminalRun(ctx, owner, runID) }
	changing := get(pinnedPath)
	if !trace.called || trace.err != nil {
		t.Fatalf("deletion interleaving called=%t err=%v", trace.called, trace.err)
	}
	if changing.Code != http.StatusConflict {
		t.Errorf("deletion during pinned provenance returned %d; want409", changing.Code)
	}
	for auditID, auditOwner := range auditOwners {
		after, err := audits.Get(ctx, auditOwner, auditID)
		if err != nil {
			t.Fatal(err)
		}
		affected := auditID == "audit-a" || auditID == "audit-b"
		if affected && (after.Revision != before[auditID].Revision+1 || !after.UpdatedAt.After(before[auditID].UpdatedAt)) {
			t.Errorf("affected Audit %s did not advance revision/updatedAt: before=%d/%s after=%d/%s", auditID, before[auditID].Revision, before[auditID].UpdatedAt, after.Revision, after.UpdatedAt)
		}
		if !affected && !reflect.DeepEqual(after, before[auditID]) {
			t.Errorf("unaffected Audit %s changed", auditID)
		}
		if auditOwner == owner {
			response := get("/v1/audits/" + auditID)
			if response.Code != http.StatusOK || (response.Header().Get("ETag") != etags[auditID]) != affected {
				t.Errorf("Audit %s ETag changed=%t status=%d; affected=%t", auditID, response.Header().Get("ETag") != etags[auditID], response.Code, affected)
			}
		}
	}
	for _, stalePath := range []string{pinnedPath, path + "?limit=1&cursor=" + url.QueryEscape(*initial.Page.NextCursor)} {
		if response := get(stalePath); response.Code != http.StatusConflict {
			t.Errorf("stale context returned%d; want409", response.Code)
		}
	}
	refreshedResponse := get(path + "?limit=200")
	var refreshed findingProvenancePageResponse
	if err := json.Unmarshal(refreshedResponse.Body.Bytes(), &refreshed); err != nil || refreshedResponse.Code != http.StatusOK || len(refreshed.Items) != 2 {
		t.Fatalf("refreshed page: %d %s err=%v", refreshedResponse.Code, refreshedResponse.Body.String(), err)
	}
	currentAudit, err := audits.Get(ctx, owner, "audit-a")
	if err != nil || refreshed.AuditRevision != currentAudit.Revision || refreshed.AuditRevision <= initial.AuditRevision {
		t.Errorf("refreshed envelope revision=%d current Audit=%d initial=%d err=%v", refreshed.AuditRevision, currentAudit.Revision, initial.AuditRevision, err)
	}
	if refreshed.FindingRevision != initial.FindingRevision {
		t.Errorf("source availability changed finding revision from %d to %d", initial.FindingRevision, refreshed.FindingRevision)
	}
	for _, item := range refreshed.Items {
		suffix := strings.TrimPrefix(item.ReceiptID, "receipt-")
		original, ok := beforeReceipts[suffix]
		if !ok || item.RecordID != "proposal:"+item.ReceiptID || item.Origin.RunID != runID || !item.Origin.RunDeleted ||
			len(original.AuditHolds) != 1 || !reflect.DeepEqual(item.Proposal.Ref, original.AuditHolds[0].Proposal.Ref) || item.Proposal.Digest != original.AuditHolds[0].Proposal.Digest {
			t.Fatalf("refreshed source identity changed: %+v", item)
		}
	}
	for suffix, auditID := range destinations {
		after, err := intake.GetAuditReceipt(ctx, owner, auditID, "receipt-"+suffix)
		if err != nil {
			t.Fatal(err)
		}
		expected := beforeReceipts[suffix]
		expected.Origin.RunDeleted = true
		if !reflect.DeepEqual(after, expected) {
			t.Fatalf("retained exact receipt %s changed beyond RunDeleted: before=%+v after=%+v", suffix, expected, after)
		}
		for _, hold := range after.AuditHolds {
			for _, evidence := range hold.Evidence {
				read, err := projectArtifacts.Read(ctx, evidence.Ref)
				if err != nil || auditHandlerDigest(string(read.Payload.Data)) != evidence.Digest {
					t.Fatalf("retained exact evidence %s unavailable: %v", suffix, err)
				}
			}
		}
	}
	if response := get("/v1/audits/foreign-audit"); response.Code != http.StatusNotFound {
		t.Fatalf("foreign Audit returned%d", response.Code)
	}
	if response := get("/v1/audits/foreign-audit/findings/" + findingID + "/provenance"); response.Code != http.StatusNotFound {
		t.Fatalf("foreign provenance returned%d", response.Code)
	}
}

func seedPublicDeletionRunReceipt(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, runID, suffix string) contracts.ArtifactRef {
	t.Helper()
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	run, err := service.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("exact evidence " + suffix)
	writtenEvidence, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: suffix}, artifacts.Payload{MediaType: "text/plain", Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evidence := findingintake.ExactArtifact{Ref: writtenEvidence.Ref, Digest: auditHandlerDigest(string(payload)), MediaType: writtenEvidence.MediaType, SizeBytes: writtenEvidence.Size}
	proposal := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: suffix, Title: suffix, Description: "A retained observation.",
		Subject: auditdomain.FindingSubject{Kind: "function", Key: suffix}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{"evidence-1"}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
	}
	body, err := auditdomain.EncodeFindingProposal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	written, err := service.WriteFindingProposal(ctx, runID, suffix, artifacts.Payload{MediaType: "application/json", Data: body})
	if err != nil {
		t.Fatal(err)
	}
	refJSON, _ := json.Marshal(written.Ref)
	evidenceJSON, _ := json.Marshal([]findingintake.ExactArtifact{evidence})
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts (
receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,
invocation_id,submission_id,client_key,request_digest,run_id,owner_id,workflow_name,workflow_version,
workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence
) VALUES($1,$2,$3,'runtime','instance','stage','worker',$4,$5,$6,$7,$8,$9,'source','1','contractor/v1alpha1',
'{"name":"source","version":"1"}',$7,$10,$11,'application/json',$12,$13)`,
		"receipt-"+suffix, "proposal-"+suffix, "allocation-"+suffix, "invocation-"+suffix, "submission-"+suffix, suffix,
		auditHandlerDigest(suffix), runID, owner, refJSON, auditHandlerDigest(string(body)), written.Size, evidenceJSON); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_retention(receipt_id,state) VALUES($1,'source-held')`, "receipt-"+suffix); err != nil {
		t.Fatal(err)
	}
	return written.Ref
}
