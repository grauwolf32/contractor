package public

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestGenericArtifactRoutesCannotExposeOrMutateAuditStandardCatalog(t *testing.T) {
	fixture := newHandlerFixture(t)
	target := contracts.ArtifactRef{
		Namespace: auditstandards.CatalogNamespace,
		Name:      "std-protected",
	}
	if _, err := fixture.artifacts.WriteAuditStandardPackage(
		t.Context(), "user-1", target,
		artifacts.Payload{MediaType: auditstandards.MediaType, Data: []byte("protected")}, nil,
	); err != nil {
		t.Fatal(err)
	}

	list := serveQuery(t, fixture.handler, "/v1/artifacts")
	if list.Code != http.StatusOK || strings.Contains(list.Body.String(), target.Name) {
		t.Fatalf("generic Artifact list exposed catalog = %d %s", list.Code, list.Body.String())
	}

	for _, suffix := range []string{"", "/metadata", "/versions", "/lineage"} {
		request := authenticatedRequest(
			http.MethodGet,
			"/v1/artifacts/"+target.Namespace+"/"+target.Name+suffix,
			bytes.NewReader(nil),
		)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound || strings.Contains(response.Body.String(), "protected") {
			t.Errorf("generic catalog read %q = %d %s", suffix, response.Code, response.Body.String())
		}
	}

	put := authenticatedRequest(
		http.MethodPut,
		"/v1/artifacts/"+target.Namespace+"/replacement",
		bytes.NewReader([]byte("replacement")),
	)
	put.Header.Set("Content-Type", "application/octet-stream")
	put.Header.Set("If-None-Match", "*")
	putResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(putResponse, put)
	if putResponse.Code != http.StatusBadRequest {
		t.Fatalf("generic catalog write = %d %s", putResponse.Code, putResponse.Body.String())
	}
}
