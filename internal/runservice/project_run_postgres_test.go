package runservice

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// readProjectBeforePurge answers the Project precheck with the Project as it
// was read before a purge removed it, so the Run insert is the first statement
// that meets the missing row.
type readProjectBeforePurge struct{ project projectstore.Project }

func (reader readProjectBeforePurge) Get(_ context.Context, ownerID, projectID string) (projectstore.Project, error) {
	if ownerID != reader.project.OwnerID || projectID != reader.project.ProjectID {
		return projectstore.Project{}, projectstore.ErrNotFound
	}
	return reader.project, nil
}

// A Project Run for a Project its owner does not have is not found (PRJ-R5).
// The precheck answers the ordinary request before any transaction begins;
// a Project that disappears after the precheck is the same not-found from
// the Run insert, and the transaction stores nothing.
func TestPostgresProjectRunForAMissingProjectIsNotFound(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	root := t.TempDir()
	for _, directory := range []string{"agent-templates", "workflows", "instructions", "model-policies", "llm-gateways", "audit-profiles"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, relative := range []string{"workflows/nuclei.yaml", "agent-templates/nuclei.yaml", "instructions/scan.md"} {
		data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "scan-catalog", relative))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, relative), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	manager, _, err := configload.NewManager(config.ManagerOptions{OperatorRoot: root, ManagedRoot: filepath.Join(t.TempDir(), "managed")})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	projects := projectstore.NewPostgresStore(pool)
	const owner = "project-run-owner"
	present, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-present", OwnerID: owner, Kind: projectstore.KindProject,
		Name: "Present", IdempotencyKey: "project-present", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	transactions := 0
	committed := postgresPublicTransaction(pool)
	transaction := func(ctx context.Context, fn func(PublicRunWriter, *artifacts.Service, RuntimeCredentialValidator) error) error {
		transactions++
		return committed(ctx, fn)
	}
	newService := func(reader ProjectReader) *Service {
		service, err := New(Options{
			Runs: runs, Workflows: manager, LLMCredentials: emptyCredentialLookup{},
			CredentialGuard: openCredentialGuard{}, RuntimeCredentials: acceptRuntimeCredentials{},
			Projects: reader, PublicTransaction: transaction,
		})
		if err != nil {
			t.Fatal(err)
		}
		return service
	}
	create := func(t *testing.T, service *Service, runID, projectID string) (CreateResult, error) {
		t.Helper()
		params := PublicCreateParams{
			OwnerID: owner, ProjectID: &projectID, Workflow: "nuclei-target@1",
			Parameters: map[string]string{"target": "https://fixture.invalid/"},
			Inputs:     map[string]contracts.ArtifactRef{}, IdempotencyKey: runID,
			NewRunID: func() (string, error) { return runID, nil },
		}
		digest, digestErr := PublicRequestDigest(params)
		if digestErr != nil {
			t.Fatal(digestErr)
		}
		params.RequestDigest = digest
		return service.CreatePublic(ctx, params)
	}
	requireNoRun := func(t *testing.T, runID string) {
		t.Helper()
		var rows int
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM workflow_runs WHERE run_id = $1)
     + (SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'run' AND scope_id = $1)`, runID,
		).Scan(&rows); err != nil {
			t.Fatal(err)
		}
		if rows != 0 {
			t.Fatalf("rejected Run %q stored %d rows", runID, rows)
		}
	}

	t.Run("present Project", func(t *testing.T) {
		result, err := create(t, newService(projects), "run-present", present.ProjectID)
		if err != nil || !result.Created || result.Run.ProjectID == nil || *result.Run.ProjectID != present.ProjectID {
			t.Fatalf("create in a present Project = (%+v, %v)", result, err)
		}
	})

	t.Run("precheck", func(t *testing.T) {
		transactions = 0
		_, err := create(t, newService(projects), "run-precheck", "project-missing")
		if !errors.Is(err, projectstore.ErrNotFound) || transactions != 0 {
			t.Fatalf("create in a missing Project = %v after %d transactions, want the precheck's ErrNotFound", err, transactions)
		}
		requireNoRun(t, "run-precheck")
	})

	t.Run("Project purged after the precheck", func(t *testing.T) {
		transactions = 0
		purged := present
		purged.ProjectID = "project-purged"
		_, err := create(t, newService(readProjectBeforePurge{project: purged}), "run-purged", purged.ProjectID)
		if !errors.Is(err, runstore.ErrNotFound) || transactions != 1 {
			t.Fatalf("create in a purged Project = %v after %d transactions, want the Run store's ErrNotFound", err, transactions)
		}
		requireNoRun(t, "run-purged")
	})
}
