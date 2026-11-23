package public

import (
	"net/http"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A Project Run for an unknown or another owner's Project keeps the
// owner-safe 404 of the existing Project precheck (PRJ-R5) on the real
// stores, and stores no Run.
func TestPostgresProjectRunForAMissingProjectAnswersNotFound(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	projects := projectstore.NewPostgresStore(pool)
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Runs, d.Artifacts, d.Projects = runs, artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)), projects
		d.RunQueue, d.RunLifecycle = runs, runs
		d.RunCreator = newTestRunCreator(t, *d, runs, integrationUnitOfWork{pool: pool, credentials: d.Credentials}, true)
	})
	if _, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-foreign", OwnerID: "user-2", Kind: projectstore.KindProject,
		Name: "Foreign", IdempotencyKey: "project-foreign", RequestDigest: "sha256:" + strings.Repeat("f", 64),
	}); err != nil {
		t.Fatal(err)
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, projectID := range []string{"project-missing", "project-foreign"} {
		request := newPublicContractRequest(http.MethodPost, "/v1/projects/"+projectID+"/runs", []byte(
			`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"sources","name":"service"}}}`,
		))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(idempotencyKeyHeader, "run-in-"+projectID)
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
		if response.Code != http.StatusNotFound {
			t.Fatalf("create Run in %q = %d %s", projectID, response.Code, response.Body.String())
		}
		assertErrorCode(t, response, "not_found")
	}
	var stored int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&stored); err != nil || stored != 0 {
		t.Fatalf("missing Project requests stored %d Runs (%v)", stored, err)
	}
}
