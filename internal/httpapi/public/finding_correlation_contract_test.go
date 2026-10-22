package public

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

type correlationContractService struct{ *fakeAuditManagement }

func correlationContractFence(audits ...string) auditstore.FindingCorrelationFence {
	value := auditstore.FindingCorrelationFence{ProjectID: "project-one", Audits: []auditstore.FindingComparisonAudit{}}
	for _, id := range audits {
		value.Audits = append(value.Audits, auditstore.FindingComparisonAudit{AuditID: id, Revision: 1, State: auditstore.AuditDraft, Profile: auditstore.FindingComparisonProfile{Name: "profile", Version: "1", Digest: auditHandlerDigest("profile")}, BaselineDigestEncoding: "postgres-jsonb-text/v1", Coverage: "not-compared"})
	}
	return value
}
func (*correlationContractService) FindingCandidates(context.Context, auditservice.FindingCandidatesParams) (auditservice.FindingCandidatesPage, error) {
	fence := correlationContractFence("audit-one")
	fence.ProjectRevision = 3
	return auditservice.FindingCandidatesPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, Scope: auditservice.FindingCorrelationScope{Kind: "project", ProjectID: "project-one", AuditIDs: []string{}}, Fence: fence, Source: auditservice.FindingCandidateSourceProjection{AuditID: "audit-one", FindingID: "finding-one", ReceiptID: "receipt-one", FirstReceiptID: "receipt-one", HintsAvailability: "unavailable", HintsReason: "not_recorded"}, TargetBasis: "immutable-first-receipt", Items: []auditservice.FindingCandidate{}}, nil
}
func (*correlationContractService) CompareFindings(context.Context, auditservice.FindingComparisonParams) (auditservice.FindingComparisonPage, error) {
	return auditservice.FindingComparisonPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, Scope: auditservice.FindingCorrelationScope{Kind: "audit-pair", ProjectID: "project-one", AuditIDs: []string{"audit-one", "audit-two"}}, Fence: correlationContractFence("audit-one", "audit-two"), LeftAuditID: "audit-one", RightAuditID: "audit-two", Limitations: []string{"absence-is-not-remediation"}, Items: []auditservice.FindingComparisonRow{}}, nil
}
func (*correlationContractService) FindingGroupMembers(context.Context, auditservice.FindingMembersParams) (auditservice.FindingMembersPage, error) {
	fence := correlationContractFence()
	fence.ProjectRevision = 3
	return auditservice.FindingMembersPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, Scope: auditservice.FindingCorrelationScope{Kind: "project", ProjectID: "project-one", AuditIDs: []string{}}, Fence: fence, GroupKey: strings.Repeat("a", 64), Representative: auditdomain.FindingCorrelationReference{AuditID: "audit-one", FindingID: "finding-one"}, Items: []auditstore.FindingComparisonPreview{}}, nil
}
func conformFindingCorrelationOperations(t *testing.T, router routers.Router) {
	t.Helper()
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(services *testServices) { services.Audits = &correlationContractService{&fakeAuditManagement{}} })
	for _, path := range []string{
		"/v1/audits/audit-one/findings/finding-one/duplicate-candidates",
		"/v1/projects/project-one/finding-comparisons?leftAuditId=audit-one&rightAuditId=audit-two",
		"/v1/projects/project-one/finding-correlations/" + strings.Repeat("a", 64) + "/members",
	} {
		contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path, ""), http.StatusOK)
	}
}

func TestFindingCorrelationCursorIsBoundedAndQueryBound(t *testing.T) {
	cursor := pageCursors{key: [32]byte{1, 2, 3}}
	position := &auditservice.FindingCorrelationPosition{ProjectRevision: 9007199254740991, AuditRevisions: []uint64{9007199254740991, 9007199254740991}, Stage: 4, Key: strings.Repeat("a", 256), CreatedAt: "2026-10-22T12:34:56.999999999Z"}
	encoded, err := cursor.encodeFindingCorrelation("one", position)
	if err != nil || encoded == nil || len(*encoded) > auditservice.FindingCorrelationMaximumCursorBytes {
		t.Fatalf("cursor bound: %v %v", encoded, err)
	}
	decoded, err := cursor.decodeFindingCorrelation(*encoded, "one")
	if err != nil || decoded.Key != position.Key || decoded.ProjectRevision != position.ProjectRevision {
		t.Fatalf("cursor roundtrip: %+v %v", decoded, err)
	}
	if _, err := cursor.decodeFindingCorrelation(*encoded, "other-query"); err == nil {
		t.Fatal("cursor crossed selection")
	}
	if _, err := cursor.decodeFindingCorrelation((*encoded)[:len(*encoded)-1], "one"); err == nil {
		t.Fatal("tampered cursor accepted")
	}
}
