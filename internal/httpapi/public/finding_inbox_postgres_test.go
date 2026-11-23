//go:build integration

package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// seedPublicInboxReceipt records owner's receipt key of Run runID with its
// exact proposal, originated by Audit auditID unless it is empty.
func seedPublicInboxReceipt(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, runID, auditID, key string) contracts.ArtifactRef {
	t.Helper()
	proposal := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: key, Title: key, Description: "An inbox observation.",
		Subject: &auditdomain.FindingSubject{Kind: "function", Key: key}, Preconditions: []string{},
		StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
	}
	body, err := auditdomain.EncodeFindingProposal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(proposal))
	if err != nil {
		t.Fatal(err)
	}
	written, err := artifactstest.WriteFindingProposal(ctx, artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)),
		runID, key, artifacts.Payload{MediaType: "application/json", Data: body})
	if err != nil {
		t.Fatal(err)
	}
	refJSON, _ := json.Marshal(written.Ref)
	var audit *string
	if auditID != "" {
		audit = &auditID
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts (
receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,
invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,
audit_execution_id,audit_id,audit_task_id,workflow_name,workflow_version,
workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,
proposal_size_bytes,evidence,report_projection
, checks_projection) VALUES($1,$2,'allocation-inbox','runtime','instance','stage','worker',$3,$4,$5,$6,$7,$8,'inbox-project',
CASE WHEN $9::text IS NOT NULL THEN $9 || '-execution' END,$9::text,CASE WHEN $9::text IS NOT NULL THEN 'discovery' END,
'source','1','contractor/v1alpha1','{"name":"source","version":"1"}',$6,$10,$11,'application/json',$12,'[]',$13, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($11)::text,'size_bytes',($12)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		"receipt-"+key, "proposal-"+key, "invocation-"+key, "submission-"+key, key, auditHandlerDigest(key),
		runID, owner, audit, refJSON, auditHandlerDigest(string(body)), written.Size, string(projection)); err != nil {
		t.Fatal(err)
	}
	return written.Ref
}

// The public Audit inbox pages through native-only, held-only and
// overlapping receipts, receipts sharing one creation time, an old receipt
// imported later and a held receipt of a deleted Run in receipt-time order.
// Each page carries exactly the service's receipts for its signed cursor, the
// look-ahead row sets hasMore, and the last page has no cursor.
func TestPostgresPublicAuditInboxPagesThroughCursors(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, projectID, auditID, siblingID = "user-1", "inbox-project", "inbox-audit", "inbox-sibling"
	const goneRun, liveRun = "inbox-gone-run", "inbox-live-run"
	for _, pair := range [][2]string{{owner, projectID}, {owner, "inbox-other-project"}} {
		if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
			OwnerID: pair[0], ProjectID: pair[1], Kind: projectstore.KindProject, Name: pair[1],
			IdempotencyKey: pair[1], RequestDigest: auditHandlerDigest(pair[1]),
		}); err != nil {
			t.Fatal(err)
		}
	}
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	input, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	selection, err := json.Marshal(auditservice.DraftSelection{Labels: map[string]string{}, RequiredPlacementLabels: []string{},
		Schema:        auditservice.DraftSelectionSchema,
		Inputs:        map[string]auditstore.ExactArtifact{"source": {Ref: input.Ref, Digest: auditHandlerDigest("source"), MediaType: input.MediaType, SizeBytes: input.Size}},
		RuntimeLabels: []string{},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{auditID, siblingID} {
		if _, _, err := auditstoretest.NewStore(pool, artifactpgtest.NewStorage()).CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: id, OwnerID: owner, ProjectID: projectID,
			Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditHandlerDigest("profile")},
			ProfileSnapshot: json.RawMessage(`{"interaction":{}}`), InputSelection: selection,
			Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20},
			IdempotencyKey: id, RequestDigest: auditHandlerDigest(id),
		}); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, _, err := configload.LoadUnionReadOnly("../../config/testdata/valid", filepath.Join(t.TempDir(), "managed"))
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
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	project := projectID
	for _, runID := range []string{goneRun, liveRun} {
		if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
			LaneClass: resolved.LaneClassModel,
			RunID:     runID, OwnerID: owner, ProjectID: &project, WorkflowName: workflow.Ref.Name,
			WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
			WorkflowSnapshot: workflowJSON, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "inbox-done"}); err != nil {
			t.Fatal(err)
		}
	}
	refs := map[string]contracts.ArtifactRef{}
	for _, seed := range [][3]string{
		{goneRun, "", "imported"}, {goneRun, auditID, "gonenative"}, {goneRun, auditID, "goneheld"},
		{liveRun, auditID, "nativea"}, {liveRun, auditID, "nativeb"}, {liveRun, auditID, "nativec"},
		{liveRun, auditID, "overlap"}, {liveRun, siblingID, "siblingheld"}, {liveRun, siblingID, "siblingonly"},
	} {
		refs[seed[2]] = seedPublicInboxReceipt(t, ctx, pool, owner, seed[0], seed[1], seed[2])
	}
	// nativeb, nativec, overlap and siblingheld share one creation time.
	for _, statement := range []string{
		`ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable`,
		`UPDATE finding_proposal_receipts SET created_at = (SELECT created_at FROM finding_proposal_receipts WHERE receipt_id = 'receipt-nativeb')
		  WHERE receipt_id IN ('receipt-nativec', 'receipt-overlap', 'receipt-siblingheld')`,
		`ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	request := func(runID, key string) findingintake.ImportRequest {
		return findingintake.ImportRequest{OwnerID: owner, AuditID: auditID, RunID: runID, Proposal: refs[key]}
	}
	if err := intake.RetainAuditCollectionBatch(ctx, []findingintake.ImportRequest{request(goneRun, "goneheld")}); err != nil {
		t.Fatal(err)
	}
	if err := intake.RetainAuditCollectionBatch(ctx, []findingintake.ImportRequest{request(liveRun, "overlap")}); err != nil {
		t.Fatal(err)
	}
	// The oldest receipt is imported last.
	for _, imported := range []findingintake.ImportRequest{request(liveRun, "siblingheld"), request(goneRun, "imported")} {
		if _, _, err := intake.ImportIntoAudit(ctx, imported); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
SELECT receipt_id, $1, 'inbox-other-project', proposal_ref, '[]'::jsonb
  FROM finding_proposal_receipts WHERE receipt_id = 'receipt-siblingonly'`, auditID); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, owner, goneRun); err != nil {
		t.Fatal(err)
	}
	expected := []string{"receipt-imported", "receipt-goneheld", "receipt-nativea", "receipt-nativeb",
		"receipt-nativec", "receipt-overlap", "receipt-siblingheld"}
	var want []string
	if err := pool.QueryRow(ctx, `
SELECT array_agg(receipt_id ORDER BY created_at, receipt_id)
  FROM finding_proposal_receipts WHERE receipt_id = ANY($1)`, expected).Scan(&want); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(deps *testServices) {
		credentials := newFakeManagedCredentials()
		audits, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(),
			Pool: pool, Profiles: deps.Config.(*config.Manager), CredentialGuard: credentials,
			TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil }),
		})
		if err != nil {
			t.Fatal(err)
		}
		deps.Audits = audits
		deps.FindingProposals = intake
	})
	for limit := 1; limit <= len(want)+1; limit++ {
		var listed []string
		query := findingintake.ListQuery{Limit: limit}
		cursor := ""
		for pages := 0; ; pages++ {
			if pages > len(want) {
				t.Fatalf("inbox at limit %d does not end", limit)
			}
			path := "/v1/audits/" + auditID + "/finding-proposals?limit=" + strconv.Itoa(limit)
			if cursor != "" {
				path += "&cursor=" + url.QueryEscape(cursor)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, path, nil).WithContext(ctx))
			var page struct {
				Items []json.RawMessage `json:"items"`
				Page  struct {
					HasMore    bool    `json:"hasMore"`
					NextCursor *string `json:"nextCursor"`
				} `json:"page"`
			}
			if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &page) != nil {
				t.Fatalf("inbox page at limit %d: %d %s", limit, response.Code, response.Body.String())
			}
			service, err := intake.ListAuditInbox(ctx, owner, auditID, query)
			if err != nil {
				t.Fatal(err)
			}
			if len(service) > limit {
				service = service[:limit]
			}
			if len(page.Items) != len(service) {
				t.Fatalf("inbox page at limit %d has %d items, service %d", limit, len(page.Items), len(service))
			}
			for index, item := range page.Items {
				encoded, err := json.Marshal(service[index])
				if err != nil {
					t.Fatal(err)
				}
				var got, wantItem any
				if json.Unmarshal(item, &got) != nil || json.Unmarshal(encoded, &wantItem) != nil || !reflect.DeepEqual(got, wantItem) {
					t.Fatalf("inbox item at limit %d = %s, service %s", limit, item, encoded)
				}
				listed = append(listed, service[index].ReceiptID)
			}
			remaining := len(want) - len(listed)
			if page.Page.HasMore != (remaining > 0) || (page.Page.NextCursor != nil) != page.Page.HasMore {
				t.Fatalf("inbox page at limit %d after %d receipts: hasMore=%t cursor=%v", limit, len(listed), page.Page.HasMore, page.Page.NextCursor)
			}
			if !page.Page.HasMore {
				break
			}
			cursor = *page.Page.NextCursor
			last := service[len(service)-1]
			query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
		}
		if !slices.Equal(listed, want) {
			t.Fatalf("public inbox at limit %d listed %v, want %v", limit, listed, want)
		}
	}
}
