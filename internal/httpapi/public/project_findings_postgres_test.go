package public

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestPublicProjectFindingPagesAndDirectCanonicalEdit(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, projectID, auditID = "user-1", "project-pages", "audit-pages"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: "Canonical pages", IdempotencyKey: "project", RequestDigest: auditHandlerDigest("project")}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstoretest.NewStore(pool, artifactpgtest.NewStorage()).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: auditID, OwnerID: owner, ProjectID: projectID, Profile: auditstore.ProfileIdentity{Name: "page-profile", Version: "1", Digest: auditHandlerDigest("profile")}, ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100, MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: "audit", RequestDigest: auditHandlerDigest("audit")}); err != nil {
		t.Fatal(err)
	}
	var service *auditservice.Service
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
		credentials := newFakeManagedCredentials()
		var err error
		service, err = auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Profiles: dependencies.Config.(*config.Manager), CredentialGuard: credentials, TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil })})
		if err != nil {
			t.Fatal(err)
		}
		intake, err := findingintake.New(pool, artifactpgtest.NewStorage())
		if err != nil {
			t.Fatal(err)
		}
		dependencies.Audits = service
		dependencies.FindingProposals = intake
	})
	var selected string
	for i := range 201 {
		id := seedPublicPaginationFinding(t, ctx, pool, owner, projectID, auditID, fmt.Sprintf("page-%03d", i))
		if i == 0 {
			selected = id
		}
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	var staleCursor string
	var sourceRevision uint64
	for _, entry := range []struct {
		path, id string
		size     int
	}{{"/v1/projects/" + projectID + "/findings", "findingId", 50}, {"/v1/audits/" + auditID + "/findings", "findingId", 50}, {"/v1/projects/" + projectID + "/finding-sources", "receiptId", 64}} {
		t.Run(entry.path, func(t *testing.T) {
			seen := map[string]bool{}
			query, _ := url.Parse(entry.path)
			values := query.Query()
			values.Set("limit", fmt.Sprint(entry.size))
			for {
				query.RawQuery = values.Encode()
				response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, query.String(), nil), http.StatusOK)
				if response.Body.Len() > 1<<20 {
					t.Fatal("metadata page exceeds 1 MiB")
				}
				var page struct {
					ProjectRevision uint64                       `json:"projectRevision"`
					Items           []map[string]json.RawMessage `json:"items"`
					Page            pageInfoResponse             `json:"page"`
				}
				if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
					t.Fatal(err)
				}
				if entry.id == "receiptId" {
					sourceRevision = page.ProjectRevision
				}
				if len(page.Items) > entry.size {
					t.Fatal("oversized page")
				}
				for _, row := range page.Items {
					var id string
					if err := json.Unmarshal(row[entry.id], &id); err != nil || id == "" || seen[id] {
						t.Fatalf("missing/repeated identity %s: %v", id, err)
					}
					seen[id] = true
				}
				if !page.Page.HasMore {
					break
				}
				if page.Page.NextCursor == nil {
					t.Fatal("missing continuation")
				}
				if entry.path == "/v1/projects/"+projectID+"/findings" && staleCursor == "" {
					staleCursor = *page.Page.NextCursor
				}
				values.Set("cursor", *page.Page.NextCursor)
			}
			if len(seen) != 201 {
				t.Fatalf("reached %d of 201 records", len(seen))
			}
		})
	}
	if sourceRevision == 0 {
		t.Fatal("missing source revision")
	}
	for _, path := range []string{"/v1/projects/" + projectID + "/finding-sources", "/v1/projects/" + projectID + "/finding-sources/receipt-page-000/history"} {
		// The source and history APIs intentionally use a smaller metadata page.
		contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path+"?limit=65", nil), http.StatusBadRequest)
	}
	pinnedSources := fmt.Sprintf("/v1/projects/%s/finding-sources?projectRevision=%d&limit=1", projectID, sourceRevision)
	contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, pinnedSources, nil), http.StatusOK)
	body := `{"verdict":"true_positive","severity":"high","rationale":"Explicit assessment from a check."}`
	request := func() *http.Request {
		return jsonContractRequest(http.MethodPatch, "/v1/projects/"+projectID+"/findings/"+selected, body, "If-Match", `"1"`, "Idempotency-Key", "canonical-edit")
	}
	first := contractCall(t, router, fixture.handler, request(), http.StatusOK)
	var result auditservice.ProjectFindingEditResult
	if err := json.Unmarshal(first.Body.Bytes(), &result); err != nil || result.Finding.Revision != 2 || result.Finding.State == nil || *result.Finding.State != auditstore.FindingConfirmed {
		t.Fatalf("canonical assessment: %+v %v", result, err)
	}
	replay := contractCall(t, router, fixture.handler, request(), http.StatusOK)
	if replay.Header().Get("Idempotency-Replayed") != "true" {
		t.Fatal("missing exact receipt replay")
	}
	for _, path := range []string{"/v1/projects/" + projectID + "/findings/" + selected, "/v1/audits/" + auditID + "/findings/" + selected} {
		response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path, nil), http.StatusOK)
		var value auditstore.ProjectFinding
		if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil || value.Revision != 2 || value.State == nil || *value.State != auditstore.FindingConfirmed {
			t.Fatalf("view disagrees with canonical state: %+v %v", value, err)
		}
	}
	// Filtering examines only one bounded membership window at a time.
	// Empty pages still carry a real continuation until every candidate was seen.
	target := "/v1/audits/" + auditID + "/findings?state=confirmed&limit=20"
	found, pages := 0, 0
	sawEmptyProgress := false
	for {
		response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, target, nil), http.StatusOK)
		var page struct {
			Items []auditstore.ProjectFindingSummary `json:"items"`
			Page  pageInfoResponse                   `json:"page"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		pages++
		found += len(page.Items)
		if len(page.Items) == 0 && page.Page.HasMore {
			sawEmptyProgress = true
		}
		if !page.Page.HasMore {
			break
		}
		if page.Page.NextCursor == nil || pages > 12 {
			t.Fatal("filtered membership window failed to progress")
		}
		target = "/v1/audits/" + auditID + "/findings?state=confirmed&limit=20&cursor=" + url.QueryEscape(*page.Page.NextCursor)
	}
	if found != 1 || pages != 11 || !sawEmptyProgress {
		t.Fatalf("filtered traversal: found=%d pages=%d emptyProgress=%t", found, pages, sawEmptyProgress)
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPatch, "/v1/projects/"+projectID+"/findings/"+selected, body, "If-Match", `"1"`, "Idempotency-Key", "stale-edit"), http.StatusPreconditionFailed)
	contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, pinnedSources, nil), http.StatusConflict)
	contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/projects/"+projectID+"/findings?cursor="+url.QueryEscape(staleCursor), nil), http.StatusConflict)
}
