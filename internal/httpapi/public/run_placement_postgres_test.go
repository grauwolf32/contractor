package public

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresPublicRunPlacementPinsAndReplaysSelection(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	details, err := runservice.NewRunDetailReader(runservice.PostgresRunDetailTransaction(pool, artifactpgtest.NewStorage()))
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Runs, d.Artifacts, d.Projects = runs, artifactService, projectstore.NewPostgresStore(pool)
		d.RunDetails = details
		d.RunQueue, d.RunLifecycle = runs, runs
		d.RunCreator = newTestRunCreator(t, *d, runs, integrationUnitOfWork{pool: pool, credentials: d.Credentials}, true)
	})
	sourceStore, err := artifactService.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	source, err := sourceStore.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("placement source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	body := map[string]any{"workflow": "artifact-copy@1", "parameters": map[string]string{}, "artifacts": map[string]contracts.ArtifactRef{"source": source.Ref}, "labels": map[string]string{"cloud": "aws", "team": "appsec"}}
	post := func(want int, validate bool) createRunResponse {
		t.Helper()
		encoded, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		request := newPublicContractRequest(http.MethodPost, "/v1/runs", encoded)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", "location-create")
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, validate)
		if response.Code != want {
			t.Fatalf("create = %d %s", response.Code, response.Body.String())
		}
		var result createRunResponse
		if want == http.StatusAccepted {
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
		}
		if want == http.StatusConflict {
			assertErrorCode(t, response, "idempotency_conflict")
			if !strings.Contains(response.Body.String(), `"retryable":false`) {
				t.Fatal("changed location was retryable")
			}
		}
		return result
	}
	created := post(http.StatusAccepted, true)
	if !slices.Equal(created.RequiredPlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("response selection = %v", created.RequiredPlacementLabels)
	}
	body["labels"] = map[string]string{"team": "appsec", "cloud": "aws"}
	if replay := post(http.StatusAccepted, true); replay.RunID != created.RunID {
		t.Fatal("equivalent request did not replay")
	}
	body["labels"] = map[string]string{"local": "lab"}
	post(http.StatusConflict, true)
	for _, invalid := range []any{nil, "cloud", map[string]string{"cloud": "", "local": ""}, map[string]string{"cloud": "*"}, map[string]string{"cloud": "aws,hetzner"}} {
		body["labels"] = invalid
		post(http.StatusBadRequest, false)
	}
	stored, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).GetRun(ctx, created.RunID)
	if err != nil || !slices.Equal(stored.RequiredPlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("stored immutable selection = %+v, %v", stored, err)
	}
	statusRequest := newPublicContractRequest(http.MethodGet, "/v1/runs/"+created.RunID, nil)
	statusResponse := serveAndValidatePublicContract(t, router, fixture.handler, statusRequest, true)
	var status runStatusResponse
	if statusResponse.Code != http.StatusOK {
		t.Fatalf("status = %d %s", statusResponse.Code, statusResponse.Body.String())
	}
	if err := json.Unmarshal(statusResponse.Body.Bytes(), &status); err != nil || !slices.Equal(status.RequiredPlacementLabels, stored.RequiredPlacementLabels) {
		t.Fatalf("status requirement = %v, %v", status.RequiredPlacementLabels, err)
	}
}
