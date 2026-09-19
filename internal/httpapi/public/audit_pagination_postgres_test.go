package public

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPublicAuditPaginationBoundary(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, projectID, auditID = "user-1", "project-pages", "audit-pages"
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject,
		Name: "Pagination boundary", IdempotencyKey: "project-pages", RequestDigest: auditHandlerDigest("project-pages"),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "page-profile", Version: "1", Digest: auditHandlerDigest("page-profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: "audit-pages", RequestDigest: auditHandlerDigest("audit-pages"),
	})
	if err != nil {
		t.Fatal(err)
	}
	var service *auditservice.Service
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *Dependencies) {
			credentials := newFakeManagedCredentials()
			service, err = auditservice.New(auditservice.Options{
				Pool: pool, Profiles: dependencies.Config.(*config.Manager), CredentialGuard: credentials,
				TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
					func(pgx.Tx) (config.CredentialLookup, error) { return credentials, nil },
				),
			})
			if err != nil {
				t.Fatal(err)
			}
			dependencies.Audits = service
		},
	)
	for index := range 201 {
		suffix := fmt.Sprintf("page-%03d", index)
		findingID := seedPublicPaginationFinding(t, ctx, pool, owner, projectID, auditID, suffix)
		_, err := service.CreateFindingReview(ctx, auditservice.CreateFindingReviewParams{
			OwnerID: owner, AuditID: auditID, FindingID: findingID, ExpectedRevision: 1,
			RequestID: "review-" + suffix, IdempotencyKey: "review-" + suffix,
			RequestDigest: auditHandlerDigest("review-" + suffix),
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	const firstFinding = "finding-receipt-page-000"
	// One source proposal plus 200 retained direct assessments exercises the
	// provenance cursor over heterogeneous records without running any Workers.
	_, err = pool.Exec(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, semantic_assessment,
    result_ref, result_digest, direct_verification, contract_ref, contract_digest
)
SELECT 'assessment-' || lpad(index::text, 3, '0'), $1, $2, 'receipt-page-000', 'supported',
       '{"namespace":"audit-results","name":"direct","revision":"r1"}'::jsonb, $3, true,
       '{"namespace":"audit-contracts","name":"direct","revision":"r1"}'::jsonb, $3
FROM generate_series(1, 200) AS index`, firstFinding, auditID, auditHandlerDigest("direct"))
	if err != nil {
		t.Fatal(err)
	}
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	paths := []struct{ name, path, identity string }{
		{"findings", "/v1/audits/" + auditID + "/findings", "findingId"},
		{"reviews", "/v1/audits/" + auditID + "/reviews", "requestId"},
		{"provenance", "/v1/audits/" + auditID + "/findings/" + firstFinding + "/provenance", "recordId"},
	}
	var continuations []string
	for _, list := range paths {
		t.Run(list.name, func(t *testing.T) {
			for _, limit := range []int{199, 200} {
				t.Run(fmt.Sprint(limit), func(t *testing.T) {
					path := fmt.Sprintf("%s?limit=%d", list.path, limit)
					seen := make(map[string]bool)
					var revision uint64
					for pageIndex := range 2 {
						response := serveAndValidatePublicContract(t, router, fixture.handler,
							newPublicContractRequest(http.MethodGet, path, nil), true)
						if response.Code != http.StatusOK {
							t.Fatalf("page = %d: %s", response.Code, response.Body.String())
						}
						var page struct {
							Items         []json.RawMessage `json:"items"`
							Page          pageInfoResponse  `json:"page"`
							AuditRevision uint64            `json:"auditRevision"`
						}
						if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
							t.Fatal(err)
						}
						want := limit
						if pageIndex == 1 {
							want = 201 - limit
						}
						if len(page.Items) != want || page.Page.HasMore != (pageIndex == 0) ||
							(page.Page.NextCursor != nil) != (pageIndex == 0) {
							t.Fatalf("page %d: items=%d page=%+v", pageIndex, len(page.Items), page.Page)
						}
						if pageIndex == 0 {
							revision = page.AuditRevision
						} else if page.AuditRevision != revision {
							t.Fatal("cursor traversal changed Audit revision")
						}
						for _, item := range page.Items {
							var fields map[string]json.RawMessage
							if err := json.Unmarshal(item, &fields); err != nil {
								t.Fatal(err)
							}
							var id string
							if err := json.Unmarshal(fields[list.identity], &id); err != nil || id == "" || seen[id] {
								t.Fatalf("missing or duplicate identity %q: %v", id, err)
							}
							seen[id] = true
							if list.name == "findings" {
								var finding auditservice.Finding
								if err := json.Unmarshal(item, &finding); err != nil {
									t.Fatal(err)
								}
								receipt := finding.FirstProposal
								if finding.FindingID != "finding-"+receipt.ReceiptID ||
									receipt.Document.ClientKey != receipt.ClientKey ||
									receipt.Proposal.Ref.Name != receipt.ClientKey || len(receipt.AuditHolds) != 1 ||
									receipt.AuditHolds[0].AuditID != auditID {
									t.Fatalf("receipt hydration lost identity, exact document or hold: %+v", finding)
								}
							}
						}
						if page.Page.NextCursor != nil {
							path = fmt.Sprintf("%s?limit=%d&cursor=%s", list.path, limit, url.QueryEscape(*page.Page.NextCursor))
							if limit == 200 {
								continuations = append(continuations, path)
							}
						}
					}
					if len(seen) != 201 {
						t.Fatalf("cursor traversal returned %d distinct records", len(seen))
					}
				})
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, list.path+"?limit=201", nil))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("public limit=201 = %d: %s", response.Code, response.Body.String())
			}
		})
	}
	// The new internal allowance must not bypass owner or revision fences.
	if _, err := service.ListFindingsPage(ctx, auditservice.FindingListParams{
		OwnerID: "other-owner", AuditID: auditID, Limit: 201,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign findings = %v", err)
	}
	if _, err := service.ListReviewsPage(ctx, auditservice.ReviewListParams{
		OwnerID: "other-owner", AuditID: auditID, Limit: 201,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign reviews = %v", err)
	}
	if _, err := service.ListFindingProvenance(ctx, auditservice.ProvenanceListParams{
		OwnerID: "other-owner", AuditID: auditID, FindingID: firstFinding, Limit: 201,
	}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign provenance = %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE audits SET revision = revision + 1 WHERE audit_id = $1`, auditID); err != nil {
		t.Fatal(err)
	}
	for _, path := range continuations {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusConflict {
			t.Fatalf("stale continuation = %d: %s", response.Code, response.Body.String())
		}
	}
}

// Seed retained proposal bytes and receipts through the same Audit-hold
// admission trigger used after import. Each finding has its own receipt, so
// the 201st row cannot be hidden by a deduplicated hydration request.
func seedPublicPaginationFinding(t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	ownerID, projectID, auditID, suffix string,
) string {
	t.Helper()
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + suffix,
		Title: "Candidate " + suffix, Description: "Retained pagination test candidate.",
		Subject:       auditdomain.FindingSubject{Kind: "component", Key: suffix},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
		SeveritySuggestion: "medium", Limitations: []string{},
	}
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	written, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "audit-proposals", Name: document.ClientKey},
		artifacts.Payload{MediaType: "application/json", Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{Ref: written.Ref, Digest: auditHandlerDigest(string(payload)),
		MediaType: written.MediaType, SizeBytes: written.Size}
	proposalJSON, _ := json.Marshal(proposal)
	refJSON, _ := json.Marshal(proposal.Ref)
	receiptID := "receipt-" + suffix
	_, err = pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key,
    request_digest, run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES ($1, $2, $3, 'runtime-pages', 'instance-pages', 'stage-pages', 'worker', $4, $5, $6,
    $7, $8, $9, $10, 'finding-source', '1', 'contractor/v1alpha1',
    '{"name":"finding-source","version":"1"}'::jsonb, $11, $12::jsonb, $13, 'application/json', $14, '[]'::jsonb)`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix, "invocation-"+suffix, "submission-"+suffix,
		document.ClientKey, auditHandlerDigest("request-"+suffix), "deleted-run-"+suffix,
		ownerID, projectID, auditHandlerDigest("workflow"), refJSON, proposal.Digest, proposal.SizeBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_retention (receipt_id, state, source_run_deleted_at)
VALUES ($1, 'audit-held', clock_timestamp())`, receiptID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
VALUES ($1, $2, $3, $4::jsonb, '[]'::jsonb)`, receiptID, auditID, projectID, proposalJSON); err != nil {
		t.Fatal(err)
	}
	return "finding-" + receiptID
}
