package auditservice

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestWebScanAuditPinsOneProjectWithoutModelOrOpenAPI(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, dsn)
	catalog := &switchableProfileCatalog{snapshot: loadAuditServiceCatalog(t), available: true}
	service, err := New(Options{Pool: pool, Profiles: catalog,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) {
			return &switchableCredentialLookup{available: true}, nil
		}),
		CredentialGuard: &countingCredentialGuard{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"installed", "custom"} {
		t.Run(mode, func(t *testing.T) {
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-web-" + mode, OwnerID: "owner-web", Kind: projectstore.KindProject, Name: "Web scan", IdempotencyKey: "project-" + mode, RequestDigest: serviceTestDigest(mode)})
			if err != nil {
				t.Fatal(err)
			}
			store, err := artifacts.NewService(artifactpostgres.NewRepository(pool)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			inputs := map[string]contracts.ArtifactRef{}
			write := func(name, media, content string) {
				t.Helper()
				value, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{MediaType: media, Data: []byte(content)}, nil)
				if err != nil {
					t.Fatal(err)
				}
				inputs[name] = value.Ref
			}
			write("target", "application/json", `{"schema":"contractor.audit.web-target.v1","url":"https://target.test/"}`)
			write("settings", "application/json", `{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"","tags":"","severity":"","rateLimit":10}`)
			if mode == "custom" {
				write("rulepack", "application/yaml", "id: fixture-rule\n")
			}
			draft, _, err := service.CreateDraft(ctx, CreateDraftParams{AuditID: "audit-web-" + mode, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: ProfileSelector{Name: "web-scan", Version: "1"}, Inputs: inputs, IdempotencyKey: "create-" + mode, RequestDigest: serviceTestDigest("create-" + mode)})
			if err != nil {
				t.Fatal(err)
			}
			params := StartParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-" + mode, RequestDigest: serviceTestDigest("start-" + mode)}
			started, err := service.Start(ctx, params)
			if err != nil || started.Audit.State != auditstore.AuditActive || len(started.Items) != 1 {
				t.Fatalf("start: %v", err)
			}
			baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
			if err != nil || len(baseline.LLMCredentialIDs) != 0 {
				t.Fatalf("model-free baseline: %v", err)
			}
			if len(baseline.Inputs) != len(inputs) {
				t.Fatal("exact rule/source inputs lost")
			}
			if started.Items[0].Kind != "web-scan" {
				t.Fatal("not a direct-target Audit item")
			}
			replay, err := service.Start(ctx, params)
			if err != nil || !replay.Replayed || replay.Audit.AuditID != started.Audit.AuditID {
				t.Fatalf("ambiguous start replay: %v", err)
			}
		})
	}
}
