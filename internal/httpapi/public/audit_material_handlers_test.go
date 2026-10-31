package public

import (
	"context"
	"net/http"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type materialContractManagement struct {
	fakeAuditManagement
	owner, project, key string
	request             auditservice.MaterialPublicationRequest
	expected            *string
	replay              bool
	calls               int
}

func (f *materialContractManagement) PublishMaterial(_ context.Context, owner, project, key string, request auditservice.MaterialPublicationRequest, expected *string) (auditservice.MaterialPublication, error) {
	f.calls++
	f.owner, f.project, f.key, f.request, f.expected = owner, project, key, request, expected
	ref := request.Target
	revision := "rev-project"
	ref.Revision = &revision
	return auditservice.MaterialPublication{ProjectID: project, AuditID: request.AuditID, Source: request.Source, Artifact: ref, Replayed: f.replay}, f.err
}

func TestSharedPublicAuditMaterialPublicationContract(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	management := &materialContractManagement{}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(deps *testServices) { deps.Audits = management })
	body := `{"auditId":"audit-one","source":{"namespace":"outputs","name":"material","revision":"rev-source"},"target":{"namespace":"materials","name":"selected"}}`
	path := "/v1/projects/project-one/audit-material-publications"
	response := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, body, "Idempotency-Key", "publish"), http.StatusCreated)
	if response.Header().Get("ETag") != `"rev-project"` || management.owner != "user-1" || management.project != "project-one" || management.key != "publish" || management.expected != nil {
		t.Fatalf("publication context: %+v headers=%v", management, response.Header())
	}
	management.replay = true
	response = contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, body, "Idempotency-Key", "publish"), http.StatusOK)
	if response.Header().Get("Idempotency-Replayed") != "true" {
		t.Fatal("replay was not identified")
	}
	management.replay = false
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, body, "Idempotency-Key", "replace", "If-Match", `"rev-before"`), http.StatusOK)
	if management.expected == nil || *management.expected != "rev-before" {
		t.Fatal("target CAS was not forwarded")
	}
	management.err = contracts.ErrIdempotencyConflict
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, body, "Idempotency-Key", "publish"), http.StatusConflict)
	before := management.calls
	invalid := body[:len(body)-1] + `,"scope":"foreign"}`
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, invalid, "Idempotency-Key", "forged"), http.StatusBadRequest)
	if management.calls != before {
		t.Fatal("unknown authority member reached service")
	}
}
