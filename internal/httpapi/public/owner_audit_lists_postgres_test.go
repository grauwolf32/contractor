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
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestPublicOwnerAuditLists(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner = "user-1"
	seed := func(projectID, auditID, ownerID string) {
		t.Helper()
		_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject, Name: projectID, IdempotencyKey: projectID, RequestDigest: auditHandlerDigest(projectID)})
		if err != nil {
			t.Fatal(err)
		}
		revision := "r1"
		selection, _ := json.Marshal(auditservice.DraftSelection{Schema: auditservice.DraftSelectionSchema, Inputs: map[string]auditstore.ExactArtifact{"source": {Ref: contracts.ArtifactRef{Namespace: "source", Name: "input", Revision: &revision}, Digest: auditHandlerDigest("input"), MediaType: "text/plain", SizeBytes: 1}}, RuntimeLabels: []string{}})
		_, _, err = auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: auditID, OwnerID: ownerID, ProjectID: projectID, Profile: auditstore.ProfileIdentity{Name: "page-profile", Version: "1", Digest: auditHandlerDigest("profile")}, ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: selection, Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: auditID, RequestDigest: auditHandlerDigest(auditID)})
		if err != nil {
			t.Fatal(err)
		}
	}
	for i := range 51 {
		seed(fmt.Sprintf("project-owner-%02d", i), fmt.Sprintf("audit-owner-%02d", i), owner)
	}
	seed("project-foreign", "audit-foreign", "user-foreign")
	seed("project-deleting", "audit-deleting", owner)
	var err error
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

	for i := range 201 {
		projectID, auditID := fmt.Sprintf("project-owner-%02d", i%2), fmt.Sprintf("audit-owner-%02d", i%2)
		suffix := fmt.Sprintf("owner-%03d", i)
		id := seedPublicPaginationFinding(t, ctx, pool, owner, projectID, auditID, suffix)
		_, err := service.CreateFindingReview(ctx, auditservice.CreateFindingReviewParams{OwnerID: owner, AuditID: auditID, FindingID: id, ExpectedRevision: 1, RequestID: "review-" + suffix, IdempotencyKey: "review-" + suffix, RequestDigest: auditHandlerDigest(suffix)})
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, pair := range []struct{ project, audit, owner string }{{"project-foreign", "audit-foreign", "user-foreign"}, {"project-deleting", "audit-deleting", owner}} {
		id := seedPublicPaginationFinding(t, ctx, pool, pair.owner, pair.project, pair.audit, pair.audit)
		_, err := service.CreateFindingReview(ctx, auditservice.CreateFindingReviewParams{OwnerID: pair.owner, AuditID: pair.audit, FindingID: id, ExpectedRevision: 1, RequestID: "review-" + pair.audit, IdempotencyKey: "review-" + pair.audit, RequestDigest: auditHandlerDigest(pair.audit)})
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: owner, ProjectID: "project-deleting", ExpectedRevision: 1}); err != nil {
		t.Fatal(err)
	}
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	for _, list := range []struct {
		path, id string
		count    int
	}{{"/v1/audits", "auditId", 51}, {"/v1/findings?verdict=unreviewed", "findingId", 201}, {"/v1/reviews?state=pending", "requestId", 201}} {
		t.Run(list.path, func(t *testing.T) {
			query, _ := url.Parse(list.path)
			values := query.Query()
			values.Set("limit", "50")
			seen := map[string]bool{}
			previousTime, previousID := "", ""
			var firstCursor string
			for {
				query.RawQuery = values.Encode()
				response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, query.String(), nil), true)
				if response.Code != http.StatusOK {
					t.Fatalf("page status=%d: %s", response.Code, response.Body.String())
				}
				var page struct {
					Items []map[string]json.RawMessage `json:"items"`
					Page  pageInfoResponse             `json:"page"`
				}
				if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
					t.Fatal(err)
				}
				for _, item := range page.Items {
					var id, auditID, createdAt string
					_ = json.Unmarshal(item[list.id], &id)
					_ = json.Unmarshal(item["auditId"], &auditID)
					_ = json.Unmarshal(item["createdAt"], &createdAt)
					if auditID == "audit-foreign" || auditID == "audit-deleting" || auditID == "audit-evaluation" || seen[id] {
						t.Fatalf("unowned/hidden/duplicate item %s in %s", id, list.path)
					}
					current, _ := time.Parse(time.RFC3339Nano, createdAt)
					previous, _ := time.Parse(time.RFC3339Nano, previousTime)
					if previousTime != "" && (current.After(previous) || (current.Equal(previous) && id >= previousID)) {
						t.Fatalf("non-descending keyset %s %s", createdAt, id)
					}
					seen[id] = true
					previousTime, previousID = createdAt, id
				}
				if !page.Page.HasMore {
					break
				}
				if page.Page.NextCursor == nil {
					t.Fatal("missing cursor")
				}
				if firstCursor == "" {
					firstCursor = *page.Page.NextCursor
				}
				values.Set("cursor", *page.Page.NextCursor)
			}
			if len(seen) != list.count {
				t.Fatalf("listed %d want %d", len(seen), list.count)
			}
			for _, other := range []string{"state=invalid", "limit=201", "cursor=bad", "auditRevision=1", "state=pending&state=pending"} {
				path := query.Path + "?" + other
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, path, nil))
				if response.Code != http.StatusBadRequest {
					t.Fatalf("invalid query %s = %d", other, response.Code)
				}
			}
			altered := query.Query()
			altered.Set("cursor", firstCursor)
			if query.Path == "/v1/audits" {
				altered.Set("state", "draft")
			} else {
				altered.Set("auditState", "draft")
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, newPublicContractRequest(http.MethodGet, query.Path+"?"+altered.Encode(), nil))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("cursor reused with changed filters=%d", response.Code)
			}
		})
	}
	for _, path := range []string{"/v1/findings?limit=200&verdict=unreviewed", "/v1/reviews?limit=200&state=pending"} {
		response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path, nil), true)
		if response.Code != http.StatusOK {
			t.Fatalf("200-row page=%d: %s", response.Code, response.Body.String())
		}
		var page struct {
			Items []json.RawMessage `json:"items"`
			Page  pageInfoResponse  `json:"page"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		if len(page.Items) != 200 || !page.Page.HasMore || page.Page.NextCursor == nil {
			t.Fatalf("missing full-page sentinel: %d %+v", len(page.Items), page.Page)
		}
	}
	foreignFindings, err := service.ListOwnerFindings(ctx, auditservice.OwnerFindingListParams{OwnerID: "user-foreign", Limit: 200})
	if err != nil || len(foreignFindings) != 1 || foreignFindings[0].AuditID != "audit-foreign" {
		t.Fatalf("foreign owner's list: %+v %v", foreignFindings, err)
	}
	foreignReviews, err := service.ListOwnerReviews(ctx, auditservice.OwnerReviewListParams{OwnerID: "user-foreign", Limit: 200})
	if err != nil || len(foreignReviews) != 1 || foreignReviews[0].AuditID != "audit-foreign" {
		t.Fatalf("foreign owner's reviews: %+v %v", foreignReviews, err)
	}

	// A different owner cannot consume a cursor even with the same signing key.
	h := &handler{}
	cursor, err := h.encodePageCursor("owner-audits:"+owner+"::", time.Now().UTC().Format(time.RFC3339Nano), "audit-owner-00")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.decodePageCursor(cursor, "owner-audits:user-foreign::", 2); !errors.Is(err, errInvalidRequest) {
		t.Fatalf("owner-bound cursor=%v", err)
	}
}
