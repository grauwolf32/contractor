//go:build integration

package public

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// Every paged public list accepts the documented limit=200 on the real
// stores. Handlers read limit+1 rows, and stores that allowed only 200
// rejected the largest documented page with 400 invalid_request.
func TestPostgresPublicPagedListsAcceptTheLargestPage(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	const owner, projectID, auditID, runID = "user-1", "project-pages", "audit-pages", "run-pages"
	projects := projectstore.NewPostgresStore(pool)
	// One more Project than a page, so the largest page needs its look-ahead row.
	for index := range pagelimit.MaxFetchRows {
		id := fmt.Sprintf("project-%03d", index)
		if index == 0 {
			id = projectID
		}
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: id, OwnerID: owner, Kind: projectstore.KindProject, Name: id,
			IdempotencyKey: id, RequestDigest: auditHandlerDigest(id),
		}); err != nil {
			t.Fatal(err)
		}
	}
	runs := runstore.NewPostgresStore(pool)
	project := projectID
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: owner, ProjectID: &project,
		WorkflowName: "artifact-copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:       map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	var input artifacts.WriteResult
	for _, seed := range []struct {
		store func(string) (artifacts.ScopedStore, error)
		id    string
		ref   contracts.ArtifactRef
	}{
		{artifactService.User, owner, contracts.ArtifactRef{Namespace: "projects", Name: "page"}},
		{artifactService.Project, projectID, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}},
		{artifactService.Run, runID, contracts.ArtifactRef{Namespace: "builder", Name: "result"}},
	} {
		store, err := seed.store(seed.id)
		if err != nil {
			t.Fatal(err)
		}
		written, err := store.Write(ctx, seed.ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("page")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if seed.ref.Namespace == "inputs" {
			input = written
		}
	}
	selection, err := json.Marshal(auditservice.DraftSelection{
		Schema: auditservice.DraftSelectionSchema,
		Inputs: map[string]auditstore.ExactArtifact{"source": {
			Ref: input.Ref, Digest: auditHandlerDigest("page"), MediaType: input.MediaType, SizeBytes: input.Size,
		}},
		RuntimeLabels: []string{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "page-profile", Version: "1", Digest: auditHandlerDigest("page-profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  selection,
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: auditID, RequestDigest: auditHandlerDigest(auditID),
	}); err != nil {
		t.Fatal(err)
	}
	findingID := seedPublicPaginationFinding(t, ctx, pool, owner, projectID, auditID, "largest-page")
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
	if err != nil {
		t.Fatal(err)
	}
	principalService, err := agentprincipal.NewService(agentprincipal.ServiceOptions{
		Pool: pool, DeletionGuard: registry,
	})
	if err != nil {
		t.Fatal(err)
	}
	principals, err := controlplane.NewPrincipalOperations(principalService, registry)
	if err != nil {
		t.Fatal(err)
	}
	findings, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *Dependencies) {
			managed := newFakeManagedCredentials()
			audits, err := auditservice.New(auditservice.Options{
				Pool: pool, Profiles: dependencies.Config.(*config.Manager), CredentialGuard: managed,
				TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
					func(pgx.Tx) (config.CredentialLookup, error) { return managed, nil },
				),
			})
			if err != nil {
				t.Fatal(err)
			}
			dependencies.Projects = projects
			dependencies.Runs, dependencies.RunQueue, dependencies.RunLifecycle = runs, runs, runs
			dependencies.RunDeletion = rundeletion.New(pool)
			dependencies.Artifacts = artifactService
			dependencies.Audits = audits
			dependencies.FindingProposals = findings
			dependencies.RuntimeAgentPrincipals = principals
			// The services in front of these repositories pass the limit
			// through unchanged; the repositories own the page validation.
			dependencies.ManagedCredentials = postgresCredentialList{managed, credentials.NewRepository(pool)}
			dependencies.RuntimeCredentials = postgresRuntimeCredentialList{
				newFakeRuntimeCredentialManagement(), credentials.NewRuntimeCredentialRepository(pool),
			}
			dependencies.RuntimeConfigs = postgresRuntimeConfigList{
				dependencies.RuntimeConfigs.(*fakeRuntimeConfigManagement), runtimeconfig.NewRepository(pool),
			}
		},
	)
	checkLargestPublicPage(t, fixture.handler, map[string]string{
		"listWorkflows":                     "/v1/workflows",
		"listProjects":                      "/v1/projects",
		"listProjectArtifacts":              "/v1/projects/" + projectID + "/artifacts",
		"listProjectArtifactVersions":       "/v1/projects/" + projectID + "/artifacts/inputs/source/versions",
		"getProjectArtifactLineage":         "/v1/projects/" + projectID + "/artifacts/inputs/source/lineage",
		"listProjectRuns":                   "/v1/projects/" + projectID + "/runs",
		"listAuditProfiles":                 "/v1/audit-profiles",
		"listProjectAudits":                 "/v1/projects/" + projectID + "/audits",
		"listOwnerAudits":                   "/v1/audits",
		"listOwnerFindings":                 "/v1/findings",
		"listOwnerReviews":                  "/v1/reviews",
		"listAuditItems":                    "/v1/audits/" + auditID + "/items",
		"listAuditCoverage":                 "/v1/audits/" + auditID + "/coverage?round=round-pages",
		"listAuditFindingProposals":         "/v1/audits/" + auditID + "/finding-proposals",
		"listAuditEvents":                   "/v1/audits/" + auditID + "/events",
		"listAuditFindings":                 "/v1/audits/" + auditID + "/findings",
		"listAuditFindingProvenance":        "/v1/audits/" + auditID + "/findings/" + findingID + "/provenance",
		"listAuditReviews":                  "/v1/audits/" + auditID + "/reviews",
		"listRunQueue":                      "/v1/queue",
		"listRuns":                          "/v1/runs",
		"listRunArtifacts":                  "/v1/runs/" + runID + "/artifacts",
		"listRunFindingProposals":           "/v1/runs/" + runID + "/finding-proposals",
		"listRunArtifactVersions":           "/v1/runs/" + runID + "/artifacts/builder/result/versions",
		"getRunArtifactLineage":             "/v1/runs/" + runID + "/artifacts/builder/result/lineage",
		"listArtifacts":                     "/v1/artifacts",
		"listArtifactVersions":              "/v1/artifacts/projects/page/versions",
		"getArtifactLineage":                "/v1/artifacts/projects/page/lineage",
		"listConfigurations":                "/v1/configurations/agent-templates",
		"listAgentTemplateWorkflowBindings": "/v1/configurations/agent-templates/artifact_builder/versions/1/workflow-bindings",
		"listRuntimeConfigs":                "/v1/operations/runtime-configs",
		"listRuntimeLabels":                 "/v1/operations/runtime-labels",
		"listRuntimeCredentials":            "/v1/operations/runtime-credentials",
		"listRuntimeAgentPrincipals":        "/v1/operations/runtime-agent-principals",
		"listCredentials":                   "/v1/operations/credentials",
		"listRuntimeAgents":                 "/v1/operations/runtime-agents",
		"listAllocations":                   "/v1/operations/allocations",
	}, "listProjects")
}

type postgresCredentialList struct {
	*fakeManagedCredentials
	repository *credentials.Repository
}

func (l postgresCredentialList) ListCredentials(ctx context.Context, after string, limit int) ([]credentials.Record, error) {
	return l.repository.ListCredentials(ctx, after, limit)
}

type postgresRuntimeCredentialList struct {
	*fakeRuntimeCredentialManagement
	repository *credentials.RuntimeCredentialRepository
}

func (l postgresRuntimeCredentialList) List(
	ctx context.Context, after string, limit int,
) ([]credentials.RuntimeCredentialMetadata, error) {
	return l.repository.ListActiveMetadata(ctx, after, limit)
}

type postgresRuntimeConfigList struct {
	*fakeRuntimeConfigManagement
	repository *runtimeconfig.Repository
}

func (l postgresRuntimeConfigList) ListVersions(
	ctx context.Context, afterName, afterVersion string, limit int,
) ([]runtimeconfig.Version, error) {
	return l.repository.ListVersions(ctx, afterName, afterVersion, limit)
}

func (l postgresRuntimeConfigList) ListBindings(ctx context.Context, after string, limit int) ([]runtimeconfig.Binding, error) {
	return l.repository.ListBindings(ctx, after, limit)
}
