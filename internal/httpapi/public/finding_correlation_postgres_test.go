package public

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestPublicFindingCorrelationReadsUseOwnedRetainedObservations(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, project = "user-1", "public-correlation-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Correlation", IdempotencyKey: "correlation", RequestDigest: auditHandlerDigest("correlation")}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"public-correlation-a", "public-correlation-b"} {
		if _, _, err := auditstoretest.NewStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: id, OwnerID: owner, ProjectID: project, Profile: auditstore.ProfileIdentity{Name: "correlation", Version: "1", Digest: auditHandlerDigest("profile")}, ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2, MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: id, RequestDigest: auditHandlerDigest(id)}); err != nil {
			t.Fatal(err)
		}
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
		credentials := newFakeManagedCredentials()
		service, err := auditservice.New(auditservice.Options{Pool: pool, Profiles: dependencies.Config.(*config.Manager), CredentialGuard: credentials, TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil })})
		if err != nil {
			t.Fatal(err)
		}
		dependencies.Audits = service
	})
	revision := "1"
	source := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}, "application/zip", []byte("source"))
	report := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "report", Revision: &revision}, "application/sarif+json", []byte("report"))
	fp := auditdomain.FindingFingerprint{Schema: auditdomain.FindingFingerprintSchema, Availability: "available", Algorithm: "contractor/producer-issue/v1", Value: strings.Repeat("a", 64), Adapter: "pinned-sarif-native/v1", RuleID: "rule/test", Repository: "https://example.test/repo.git", Commit: strings.Repeat("a", 40), SourceDigest: source.Digest, Source: &source, Report: &report, RecordLocator: "/runs/0/results/0"}
	first := seedPublicPaginationFinding(t, ctx, pool, owner, project, "public-correlation-a", "public-correlation-first", fp)
	second := seedPublicPaginationFinding(t, ctx, pool, owner, project, "public-correlation-b", "public-correlation-second", fp)
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	comparePath := "/v1/projects/" + project + "/finding-comparisons?leftAuditId=public-correlation-a&rightAuditId=public-correlation-b&limit=1"
	response := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, comparePath, ""), http.StatusOK)
	var compared auditservice.FindingComparisonPage
	if err := json.Unmarshal(response.Body.Bytes(), &compared); err != nil || len(compared.Items) != 1 || compared.Items[0].Kind != "both" || compared.Items[0].PreviewsShareReceipt {
		t.Fatalf("independent comparison=%+v %v", compared, err)
	}
	group := compared.Items[0].GroupKey
	candidatePath := "/v1/audits/public-correlation-a/findings/" + first + "/duplicate-candidates"
	response = contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, candidatePath, ""), http.StatusOK)
	var candidates auditservice.FindingCandidatesPage
	if err := json.Unmarshal(response.Body.Bytes(), &candidates); err != nil || len(candidates.Items) != 1 || candidates.Items[0].Finding.FindingID != second || !candidates.Items[0].AutomaticallyCorrelated {
		t.Fatalf("candidates=%+v %v", candidates, err)
	}
	memberPath := "/v1/projects/" + project + "/finding-correlations/" + group + "/members?limit=1"
	response = contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, memberPath, ""), http.StatusOK)
	var members auditservice.FindingMembersPage
	if err := json.Unmarshal(response.Body.Bytes(), &members); err != nil || len(members.Items) != 1 || members.Page.NextCursor == nil {
		t.Fatalf("member continuation=%+v %v", members, err)
	}
	// The cursor binds its selection and limit, and result changes stale it.
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, memberPath+"&cursor="+*members.Page.NextCursor+"&auditId=public-correlation-a", ""), http.StatusBadRequest)
	if _, err := pool.Exec(ctx, `UPDATE audit_findings SET updated_at=updated_at+interval '1 microsecond' WHERE finding_id=$1`, second); err != nil {
		t.Fatal(err)
	}
	stale := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, memberPath+"&cursor="+*members.Page.NextCursor, ""), http.StatusConflict)
	var problem errorResponse
	if err := json.Unmarshal(stale.Body.Bytes(), &problem); err != nil || problem.Code != "finding_correlation_stale" {
		t.Fatalf("stale error=%+v %v", problem, err)
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, candidatePath+"?sourceReceiptId=receipt-public-correlation-second", ""), http.StatusNotFound)
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, candidatePath+"?limit=101", ""), http.StatusBadRequest)
}
