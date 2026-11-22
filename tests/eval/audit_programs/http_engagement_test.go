package auditprograms

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestWSTGHTTPProgramsStartWithTrustedEngagement(t *testing.T) {
	catalog, bundles, err := configload.LoadUnionReadOnly(configRoot(), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"owasp-wstg-4-2-active-http", "owasp-wstg-4-2-fast-active-http"} {
		t.Run(name, func(t *testing.T) {
			profile, err := catalog.AuditProfile(name + "@4")
			if err != nil {
				t.Fatal(err)
			}
			if !profile.Program.Inputs["engagement"].Required || !profile.Workflows["check"].Inputs["engagement"].Required ||
				!slices.Equal(profile.Program.Inputs["engagement"].MediaTypes, []string{engagementcontract.MediaType}) ||
				!slices.Equal(profile.Workflows["check"].Inputs["engagement"].MediaTypes, []string{engagementcontract.MediaType}) {
				t.Fatal("profile and Workflow must require typed engagement without a generic JSON alternative")
			}
			ctx := t.Context()
			pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
			if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
				t.Fatal(err)
			}
			storage := artifactpgtest.NewStorage()
			art := artifacts.NewService(storage.Repository(pool))
			standards, err := auditstandards.NewCatalog(art)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := standards.Initialize(ctx, "owner", bundles.Standards); err != nil {
				t.Fatal(err)
			}
			gateway, err := catalog.LLMGateway("local-litellm@1")
			if err != nil {
				t.Fatal(err)
			}
			now := time.Date(2026, 11, 4, 12, 0, 0, 0, time.UTC)
			service, err := auditservice.New(auditservice.Options{Storage: storage, Pool: pool, Profiles: catalog,
				TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) {
					return httpProgramCredential{gateway.Ref}, nil
				}), CredentialGuard: httpProgramGuard{}, Now: func() time.Time { return now }})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "http-project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "HTTP", ActiveChecks: contracts.ActiveChecksAutomatic, IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))}); err != nil {
				t.Fatal(err)
			}
			store, _ := art.Project("http-project")
			materials, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "materials"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("Selected fixture documentation")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			requests := uint32(7)
			var targetRequests atomic.Int32
			target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { targetRequests.Add(1) }))
			t.Cleanup(target.Close)
			document := engagementcontract.Document{Schema: engagementcontract.Schema, Egress: engagementcontract.Egress{Routes: []engagementcontract.Route{{Origin: target.URL, PathPrefixes: []string{"/api"}, Methods: []string{"GET"}, AddressRanges: []string{"127.0.0.0/8"}}}, Exclusions: []engagementcontract.Route{}},
				Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{target.URL}}},
				Limits:     &engagementcontract.HTTPLimits{RequestsPerSecond: &requests}}
			data, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			for _, mediaType := range []string{"application/json", engagementcontract.MediaType} {
				id := "typed"
				if mediaType == "application/json" {
					id = "untyped"
				}
				engagement, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: id}, artifacts.Payload{MediaType: mediaType, Data: data}, nil)
				if err != nil {
					t.Fatal(err)
				}
				draft, _, err := service.CreateDraft(ctx, auditservice.CreateDraftParams{AuditID: id, OwnerID: "owner", ProjectID: "http-project", Profile: auditservice.ProfileSelector{Name: name, Version: "4"}, Inputs: map[string]contracts.ArtifactRef{"materials": materials.Ref, "engagement": engagement.Ref}, RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Check selected WSTG assertions"}, IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id))})
				if mediaType == "application/json" {
					if !errors.Is(err, auditservice.ErrInvalid) {
						t.Fatalf("generic JSON engagement admission: %v", err)
					}
					continue
				}
				if err != nil {
					t.Fatal(err)
				}
				params := auditservice.StartParams{OwnerID: "owner", AuditID: id, ExpectedRevision: draft.Revision, IdempotencyKey: "start", RequestDigest: contentdigest.Bytes([]byte("start"))}
				started, err := service.Start(ctx, params)
				if err != nil {
					t.Fatal(err)
				}
				baseline, err := auditservice.DecodeBaseline(started.Audit.BaselineSnapshot)
				if err != nil {
					t.Fatal(err)
				}
				if baseline.Engagement == nil || !baseline.Engagement.Artifact.Ref.SameExact(engagement.Ref) || !reflect.DeepEqual(baseline.Engagement.Document, document) || baseline.Engagement.AuthorizedBy != "owner" || !baseline.Engagement.AuthorizedAt.Equal(now) {
					t.Fatal("Start did not pin the exact authorized engagement routes, identities and HTTP limits")
				}
				if started.Audit.Limits.MaxSubmittedRuns != profile.Program.Limits.MaxSubmittedRuns || started.Audit.Limits.MaxEvidenceBytes != profile.Program.Limits.MaxEvidenceBytes || started.Audit.Limits.MaxItemRunAttempts != 1 || started.Audit.DeadlineAt == nil || !started.Audit.DeadlineAt.Equal(now.Add(time.Duration(profile.Program.Limits.DeadlineSeconds)*time.Second)) || baseline.InitialAssertions == nil {
					t.Fatal("Start lost ordinary Audit limits or standard assertion source")
				}
				replay, err := service.Start(ctx, params)
				if err != nil || !replay.Replayed || string(replay.Audit.BaselineSnapshot) != string(started.Audit.BaselineSnapshot) {
					t.Fatalf("Start replay changed trusted authority: %v", err)
				}
				if targetRequests.Load() != 0 {
					t.Fatal("Start or its replay issued target HTTP requests")
				}
			}
		})
	}
}

type httpProgramCredential struct {
	gateway llmgateway.LLMGatewayConfigRef
}

func (c httpProgramCredential) LookupLLMCredential(_ context.Context, id string) (resolved.CredentialMetadata, error) {
	return resolved.CredentialMetadata{Ref: llmgateway.LLMCredentialRef{CredentialID: id}, LLMGateway: c.gateway, Unrestricted: true}, nil
}

type httpProgramGuard struct{}

func (httpProgramGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}
