package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/pentestcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type engagementGuard struct{ barrier *credentials.LifecycleBarrier }

func (g engagementGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	return g.barrier.WithCredentialReferences(ctx, fn)
}

type engagementFixture struct {
	ctx         context.Context
	pool        *pgxpool.Pool
	service     *Service
	credentials *credentials.RuntimeCredentialService
	project     projectstore.Project
	store       artifacts.ScopedStore
	checklist   contracts.ExactArtifact
	now         time.Time
}

func newEngagementFixture(t *testing.T, connections ...int32) *engagementFixture {
	return newEngagementFixtureWithEdits(t, nil, nil, connections...)
}

func newEngagementFixtureWithEdits(t *testing.T, editWorker func(string) string, customize func(map[string]string), connections ...int32) *engagementFixture {
	t.Helper()
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	pool := isolatedAuditServicePool(t, ctx, dsn)
	if len(connections) > 0 {
		config := pool.Config().Copy()
		config.MaxConns = connections[0]
		config.MinConns = 0
		config.MinIdleConns = 0
		limited, err := pgxpool.NewWithConfig(ctx, config)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(limited.Close)
		pool = limited
	}
	snapshot := loadAuditServiceProfilesWithEdits(t, func(s string) string {
		s = strings.Replace(s, "  inputs:\n", "  inputs:\n    engagement: {required: true, mediaTypes: [application/vnd.contractor.pentest-engagement+json]}\n", 1)
		return strings.Replace(s, "activeChecks: prohibited", "activeChecks: automatic", 1)
	}, editWorker, func(files map[string]string) {
		if customize != nil {
			customize(files)
		}
	})
	gateway, err := snapshot.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	barrier := credentials.NewLifecycleBarrier()
	now := time.Date(2026, 10, 28, 10, 0, 0, 0, time.UTC)
	service, err := New(Options{Pool: pool, Profiles: &switchableProfileCatalog{snapshot: snapshot, available: true}, TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil }), CredentialGuard: engagementGuard{barrier}, Now: func() time.Time { return now }})
	if err != nil {
		t.Fatal(err)
	}
	cipher, err := credentials.NewTokenCipher([]byte(strings.Repeat("e", 32)))
	if err != nil {
		t.Fatal(err)
	}
	creds, err := credentials.NewRuntimeCredentialService(credentials.RuntimeCredentialServiceOptions{Pool: pool, Lease: leasetest.Hold(t, pool).Lease, Cipher: cipher, Usage: credentials.NewRuntimeCredentialRepository(pool), Audits: hookstore.NewCredentialHolds(pool), Barrier: barrier})
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "engagement-project", OwnerID: "engagement-owner", Kind: projectstore.KindProject, Name: "Engagement", IdempotencyKey: "engagement-project", RequestDigest: serviceTestDigest("engagement-project")})
	if err != nil {
		t.Fatal(err)
	}
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, store, "engagement-checks", "automatic")
	meta, err := store.Metadata(ctx, checklist.Ref)
	if err != nil {
		t.Fatal(err)
	}
	return &engagementFixture{ctx, pool, service, creds, project, store, contracts.ExactArtifact{Ref: meta.Ref, Digest: meta.Digest, MediaType: meta.MediaType, SizeBytes: meta.Size}, now}
}
func (f *engagementFixture) document(origin string) pentestcontract.Document {
	return pentestcontract.Document{Schema: pentestcontract.Schema, AccessMode: "blackbox", Target: pentestcontract.Target{Origins: []string{origin}, SeedURLs: []string{origin + "/"}}, Scope: pentestcontract.Scope{Routes: []pentestcontract.Route{{Origin: origin, PathPrefixes: []string{"/"}, Methods: []string{"GET"}, ActionClasses: []string{"read"}, AddressRanges: []string{"127.0.0.0/8"}}}, Exclusions: []pentestcontract.Route{}, ActionClasses: []string{"read"}, CallbackDestinations: []string{}}, Identities: []pentestcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{origin}}}, Inputs: []pentestcontract.Input{{ID: "checklist", Role: "documentation", Artifact: f.checklist.Ref, ContentDigest: f.checklist.Digest}}, Policy: pentestcontract.Policy{Version: "1", ActiveChecks: "automatic", RetryPolicy: "conservative@1", ReplayPolicy: "fresh-session@1", RequiredCapabilities: []string{pentestcontract.Capability}, OptionalCapabilities: []string{}}, TestData: []pentestcontract.TestData{}, Budgets: pentestcontract.Budgets{ActiveSeconds: 120, TargetRequests: 10, RequestsPerSecond: 1, ConcurrentRequests: 1, CrawlDepth: 1, FetchedPages: 10, SurfaceVariants: 10, Hypotheses: 10, Rounds: 1, SubmittedRuns: 3, RecipeSteps: 4, ReplayExecutions: 1, LoginActions: 2, LoginAttempts: 1, RequestBodyBytes: 1024, ResponseBodyBytes: 1024, EvidenceBytes: 1024, ModelCalls: 2, ModelTokens: 4096, ActionIntents: 10}}
}
func (f *engagementFixture) create(t *testing.T, id string, document pentestcontract.Document) (auditstore.Audit, error) {
	t.Helper()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: id}, artifacts.Payload{MediaType: pentestcontract.MediaType, Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := f.service.CreateDraft(f.ctx, CreateDraftParams{AuditID: id, OwnerID: f.project.OwnerID, ProjectID: f.project.ProjectID, Profile: ProfileSelector{Name: "test-checklist", Version: "1"}, Inputs: map[string]contracts.ArtifactRef{"engagement": artifact.Ref, "checklist": f.checklist.Ref}, RuntimeLabels: []string{}, Scope: Scope{Objective: "Check fixture"}, IdempotencyKey: id, RequestDigest: serviceTestDigest(id)})
	return draft, err
}
func (f *engagementFixture) start(draft auditstore.Audit) StartParams {
	return StartParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-" + draft.AuditID, RequestDigest: serviceTestDigest("start-" + draft.AuditID)}
}
func (f *engagementFixture) credential(t *testing.T, id, owner string) {
	t.Helper()
	material, err := credentials.NewTargetBearerCredential("private-target-secret-canary")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.credentials.Create(f.ctx, credentials.RuntimeCredentialCreateRequest{CredentialID: id, ActorID: owner, IdempotencyKey: id, Material: material}); err != nil {
		t.Fatal(err)
	}
}

func TestEngagementPreflightStartPinsAndReplay(t *testing.T) {
	f := newEngagementFixture(t)
	var requests atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	defer target.Close()
	document := f.document(target.URL)
	f.credential(t, "target-one", f.project.OwnerID)
	document.Identities = append(document.Identities, pentestcontract.Identity{ID: "account-a", Kind: "bearer", Origins: []string{target.URL}, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "target-one", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-valid", document)
	if err != nil {
		t.Fatal(err)
	}
	var beforeWrites int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&beforeWrites); err != nil {
		t.Fatal(err)
	}
	preflight, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision})
	if err != nil || !preflight.ServerCompatible || len(preflight.MissingCapabilities) != 0 || !slices.Equal(preflight.CompatibleCapabilities, []string{pentestcontract.Capability}) {
		t.Fatalf("preflight=%+v err=%v", preflight, err)
	}
	var afterWrites int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&afterWrites); err != nil {
		t.Fatal(err)
	}
	if beforeWrites != afterWrites || requests.Load() != 0 {
		t.Fatalf("preflight wrote artifacts or contacted target: writes %d/%d requests %d", beforeWrites, afterWrites, requests.Load())
	}
	if holds, err := hookstore.NewCredentialHolds(f.pool).ListHeldAuditIDsByRuntimeCredential(f.ctx, "target-one", 10); err != nil || len(holds) != 0 {
		t.Fatalf("preflight retained credential: %v %v", holds, err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	// Every stored optional engagement member is omitted or non-null.
	for _, bad := range []string{`null`, `{"schema":"contractor.audit.pentest-baseline.v1","artifact":null}`} {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(started.Audit.BaselineSnapshot, &fields); err != nil {
			t.Fatal(err)
		}
		fields["engagement"] = json.RawMessage(bad)
		data, _ := json.Marshal(fields)
		if _, err := DecodeBaseline(data); !errors.Is(err, ErrInvalid) {
			t.Fatalf("invalid stored engagement accepted: %s %v", bad, err)
		}
	}
	if baseline.Engagement == nil || baseline.Engagement.AuthorizedBy != f.project.OwnerID || !baseline.Engagement.AuthorizedAt.Equal(f.now) || !slices.Contains(baseline.RuntimeCredentialIDs, "target-one") || baseline.ProjectHTTPTarget != nil {
		t.Fatalf("incorrect engagement baseline: %+v", baseline.Engagement)
	}
	if started.Audit.Limits.MaxSubmittedRuns != 3 || started.Audit.Limits.MaxEvidenceBytes != 1024 || started.Audit.DeadlineAt == nil || !started.Audit.DeadlineAt.Equal(f.now.Add(120*time.Second)) {
		t.Fatalf("finite policy not pinned: %+v", started.Audit)
	}
	if _, err := f.credentials.Delete(f.ctx, "target-one", f.project.OwnerID); !errors.Is(err, credentials.ErrRuntimeCredentialInUse) {
		t.Fatalf("held target delete=%v", err)
	}
	replay, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil || !replay.Replayed || string(replay.Audit.BaselineSnapshot) != string(started.Audit.BaselineSnapshot) {
		t.Fatalf("start replay changed authority: %v", err)
	}
	if _, err = f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision}); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("active preflight=%v", err)
	}
	if requests.Load() != 0 || strings.Contains(string(started.Audit.BaselineSnapshot), "private-target-secret-canary") {
		t.Fatal("start executed target work or exposed secret")
	}
}

func TestEngagementPreflightStaleOwnershipAndStartRevalidation(t *testing.T) {
	f := newEngagementFixture(t)
	document := f.document("http://127.0.0.1")
	f.credential(t, "target-stale", f.project.OwnerID)
	document.Identities = append(document.Identities, pentestcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Target.Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "target-stale", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-stale", document)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		owner string
		rev   uint64
		want  error
	}{{"foreign", draft.Revision, auditstore.ErrNotFound}, {f.project.OwnerID, draft.Revision + 1, auditstore.ErrPrecondition}} {
		if _, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: test.owner, AuditID: draft.AuditID, ExpectedRevision: test.rev}); !errors.Is(err, test.want) {
			t.Fatalf("owner/revision check=%v want %v", err, test.want)
		}
	}
	if _, err = f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision}); err != nil {
		t.Fatal(err)
	}
	if _, err = f.credentials.Delete(f.ctx, "target-stale", f.project.OwnerID); err != nil {
		t.Fatal(err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("start trusted stale preflight: %v", err)
	}
	current, err := f.service.Get(f.ctx, f.project.OwnerID, draft.AuditID)
	if err != nil || current.State != auditstore.AuditDraft || current.BaselineSnapshot != nil {
		t.Fatalf("failed start left baseline: %+v %v", current, err)
	}
}

func TestEngagementUnsupportedCapabilitiesAndBlackboxClosures(t *testing.T) {
	f := newEngagementFixture(t)
	document := f.document("http://127.0.0.1")
	document.Policy.RequiredCapabilities = append(document.Policy.RequiredCapabilities, "pentest-http@1")
	draft, err := f.create(t, "engagement-unsupported", document)
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision})
	if err != nil || result.ServerCompatible || !slices.Contains(result.MissingCapabilities, "pentest-http@1") {
		t.Fatalf("unimplemented transport advertised: %+v %v", result, err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("unsupported start=%v", err)
	}
	profile, err := f.service.profiles.AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	selected := &engagementSelection{document: f.document("http://127.0.0.1"), inputs: []auditstore.ArtifactLink{}}
	for _, test := range []struct {
		name   string
		mutate func(*auditprofile.ResolvedAuditProfile, *engagementSelection)
	}{
		{"unclassified inherited input", func(_ *auditprofile.ResolvedAuditProfile, e *engagementSelection) { e.unclassifiedInputs = true }},
		{"repository hydration", func(p *auditprofile.ResolvedAuditProfile, _ *engagementSelection) {
			for role, binding := range p.Workflows {
				for name, stage := range binding.Workflow.Stages {
					stage.Context.Workspace = &resolved.WorkspaceContext{Mode: contracts.WorkspaceModeDirect, Sources: []resolved.WorkspaceSource{{Artifact: "repository"}}}
					binding.Workflow.Stages[name] = stage
				}
				p.Workflows[role] = binding
			}
		}},
		{"source skill", func(p *auditprofile.ResolvedAuditProfile, _ *engagementSelection) {
			for role, binding := range p.Workflows {
				for name, stage := range binding.Workflow.Stages {
					for key, agent := range stage.Agents {
						agent.Template.Skills = []contracts.ArtifactRef{{Namespace: "skills", Name: "repository-context"}}
						stage.Agents[key] = agent
					}
					binding.Workflow.Stages[name] = stage
				}
				p.Workflows[role] = binding
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			copyProfile := profile.Clone()
			copySelection := *selected
			test.mutate(&copyProfile, &copySelection)
			_, _, reasons := engagementCompatibility(copyProfile, contracts.ActiveChecksAutomatic, &copySelection)
			if !slices.Contains(reasons, ReasonPentestSourceClosureUnsupported) {
				t.Fatalf("source closure allowed: %v", reasons)
			}
		})
	}
}

func TestEngagementSourceAssistedRetainsExactInputsAndPurgeReleasesHolds(t *testing.T) {
	f := newEngagementFixture(t)
	document := f.document("http://127.0.0.1")
	source, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original source snapshot")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := f.store.Metadata(f.ctx, source.Ref)
	if err != nil {
		t.Fatal(err)
	}
	document.AccessMode = "source-assisted"
	document.Inputs = append(document.Inputs, pentestcontract.Input{ID: "source", Role: "source", Artifact: source.Ref, ContentDigest: metadata.Digest})
	f.credential(t, "source-account", f.project.OwnerID)
	document.Identities = append(document.Identities, pentestcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Target.Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "source-account", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-source", document)
	if err != nil {
		t.Fatal(err)
	}
	if result, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision}); err != nil || !result.ServerCompatible {
		t.Fatalf("source preflight=%+v %v", result, err)
	}
	if _, err = f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("replacement source")}, source.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	held, err := f.service.ReadArtifact(f.ctx, f.project.OwnerID, draft.AuditID, "pentest-input/source", *source.Ref.Revision)
	if err != nil || string(held.Payload.Data) != "original source snapshot" {
		t.Fatalf("exact source replaced: %s %v", held.Payload.Data, err)
	}
	deleted, err := f.service.Delete(f.ctx, MutationParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "delete-source-audit", RequestDigest: serviceTestDigest("delete-source-audit")})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(f.pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claim := claimAuditForPurge(t, f.ctx, store, draft.AuditID)
	if deleted.Audit.State == auditstore.AuditCancelling {
		if settled, err := store.SettleUndispatched(f.ctx, claim, auditstore.MaxReconcileRows); err != nil || settled != 1 {
			t.Fatalf("settle undispatched engagement: %d %v", settled, err)
		}
		current, released, err := store.ReleaseDispatchHold(f.ctx, claim)
		if err != nil || !released {
			t.Fatalf("release drained engagement hold: %t %v", released, err)
		}
		if _, err = store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: current.Revision, ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting, Reason: current.StopReason}); err != nil {
			t.Fatal(err)
		}
	}
	if err = store.PurgeClaimed(f.ctx, claim, auditdomain.ArtifactNamespace(draft.AuditID)); err != nil {
		t.Fatal(err)
	}
	if _, err = f.credentials.Delete(f.ctx, "source-account", f.project.OwnerID); err != nil {
		t.Fatalf("purge retained credential: %v", err)
	}
	if _, err = f.service.Get(f.ctx, f.project.OwnerID, draft.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("purged audit exists: %v", err)
	}
}

func TestEngagementCredentialOwnershipAndStartDeleteRace(t *testing.T) {
	for _, conns := range []int32{1, 4} {
		for _, order := range []string{"start-first", "delete-first", "concurrent"} {
			t.Run(fmt.Sprintf("connections-%d/%s", conns, order), func(t *testing.T) {
				f := newEngagementFixture(t, conns)
				document := f.document("http://127.0.0.1")
				f.credential(t, "race-target", f.project.OwnerID)
				document.Identities = append(document.Identities, pentestcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Target.Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "race-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
				draft, err := f.create(t, "engagement-race", document)
				if err != nil {
					t.Fatal(err)
				}
				var startErr, deleteErr error
				start := func() { _, startErr = f.service.Start(f.ctx, f.start(draft)) }
				remove := func() { _, deleteErr = f.credentials.Delete(f.ctx, "race-target", f.project.OwnerID) }
				switch order {
				case "start-first":
					start()
					remove()
				case "delete-first":
					remove()
					start()
				default:
					ready := make(chan struct{})
					var group sync.WaitGroup
					group.Add(2)
					go func() { defer group.Done(); <-ready; start() }()
					go func() { defer group.Done(); <-ready; remove() }()
					close(ready)
					group.Wait()
				}
				if startErr == nil {
					if !errors.Is(deleteErr, credentials.ErrRuntimeCredentialInUse) {
						t.Fatalf("committed baseline lost credential: %v", deleteErr)
					}
				} else if deleteErr != nil || !errors.Is(startErr, credentials.ErrRuntimeCredentialNotFound) {
					t.Fatalf("race start=%v delete=%v", startErr, deleteErr)
				}
			})
		}
	}
	t.Run("foreign and wrong kind", func(t *testing.T) {
		f := newEngagementFixture(t)
		f.credential(t, "foreign-target", "another-owner")
		doc := f.document("http://127.0.0.1")
		doc.Identities = append(doc.Identities, pentestcontract.Identity{ID: "account-a", Kind: "bearer", Origins: doc.Target.Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "foreign-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
		if _, err := f.create(t, "engagement-foreign", doc); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
			t.Fatalf("foreign credential error=%v", err)
		}
		doc.Identities[1].Kind = "basic"
		doc.Identities[1].Credential.Kind = runtimesettings.RuntimeCredentialTargetBasic
		if _, err := f.create(t, "engagement-kind", doc); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
			t.Fatalf("wrong credential kind error=%v", err)
		}
	})
}

func TestEngagementPersistedClosureStopsBeforeExposure(t *testing.T) {
	for _, test := range []struct {
		name     string
		worker   func(string) string
		document func(*pentestcontract.Document)
		want     CompatibilityReason
	}{
		{"unclassified-selected-input", nil, func(d *pentestcontract.Document) { d.Inputs = []pentestcontract.Input{} }, ReasonPentestSourceClosureUnsupported},
		{"source-skill", func(s string) string {
			return strings.Replace(s, "  toolsets:", "  skills: [{namespace: skills, name: source-context}]\n  toolsets:", 1)
		}, nil, ReasonPentestSourceClosureUnsupported},
		{"repository-hydration", nil, nil, ReasonPentestSourceClosureUnsupported},
		{"network-sandbox", func(s string) string { return strings.Replace(s, "local-workdir@1", "podman-network@1", 1) }, nil, ReasonPentestCapabilityUnsupported},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newEngagementFixtureWithEdits(t, test.worker, func(files map[string]string) {
				if test.name == "network-sandbox" || test.name == "repository-hydration" {
					files["workflows/check.yaml"] = strings.Replace(files["workflows/check.yaml"], "      context:\n", "      context:\n        workspace:\n          mode: direct\n          sources: [{artifact: task, target: \"\"}]\n", 1)
				}
			})
			document := f.document("http://127.0.0.1")
			if test.document != nil {
				test.document(&document)
			}
			draft, err := f.create(t, "closure-"+test.name, document)
			if err != nil {
				t.Fatal(err)
			}
			preflight, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision})
			if err != nil || preflight.ServerCompatible || !slices.Contains(preflight.Reasons, test.want) {
				t.Fatalf("persisted closure: %+v %v", preflight, err)
			}
			var before int
			if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&before); err != nil {
				t.Fatal(err)
			}
			if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("source/target closure reached materialization: %v", err)
			}
			var after int
			if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&after); err != nil {
				t.Fatal(err)
			}
			if after != before {
				t.Fatalf("unsupported closure wrote staged material: %d/%d", before, after)
			}
			current, err := f.service.Get(f.ctx, f.project.OwnerID, draft.AuditID)
			if err != nil || current.State != auditstore.AuditDraft || current.BaselineSnapshot != nil {
				t.Fatalf("unsupported closure exposed baseline: %+v %v", current, err)
			}
		})
	}
}

func TestEngagementPreflightRepeatsProjectPolicyAndSuppressesDefaultIdentity(t *testing.T) {
	f := newEngagementFixture(t)
	material, err := credentials.NewHTTPOriginBearerCredential("private-default-target-canary")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.credentials.Create(f.ctx, credentials.RuntimeCredentialCreateRequest{CredentialID: "default-target", ActorID: f.project.OwnerID, IdempotencyKey: "default-target", Material: material}); err != nil {
		t.Fatal(err)
	}
	project, err := projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: f.project.ProjectID, OwnerID: f.project.OwnerID, ExpectedRevision: f.project.Revision, Name: f.project.Name, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{URL: "http://127.0.0.1", Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "default-target", Kind: runtimesettings.RuntimeCredentialOriginBearer}}})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := f.create(t, "project-policy", f.document("http://127.0.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	before, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision})
	if err != nil || !before.ServerCompatible {
		t.Fatalf("preflight before policy: %+v %v", before, err)
	}
	prohibited := contracts.ActiveChecksProhibited
	project, err = projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision, Name: project.Name, HTTPTarget: project.HTTPTarget, ActiveChecks: &prohibited})
	if err != nil {
		t.Fatal(err)
	}
	after, err := f.service.Preflight(f.ctx, PreflightParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision})
	if err != nil || after.ServerCompatible || before.BaselineInputDigest == after.BaselineInputDigest {
		t.Fatalf("stale policy preflight: %+v %v", after, err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("start trusted preflight policy: %v", err)
	}
	automatic := contracts.ActiveChecksAutomatic
	if _, err = projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision, Name: project.Name, HTTPTarget: project.HTTPTarget, ActiveChecks: &automatic}); err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil || baseline.ProjectHTTPTarget != nil || slices.Contains(baseline.RuntimeCredentialIDs, "default-target") {
		t.Fatalf("inherited mutable default identity: %+v %v", baseline, err)
	}
}

func TestEngagementNestedInputBound(t *testing.T) {
	f := newEngagementFixture(t)
	for _, count := range []int{auditprofile.MaxAuditProfileInputs, auditprofile.MaxAuditProfileInputs + 1} {
		document := f.document("http://127.0.0.1")
		for len(document.Inputs) < count {
			input := document.Inputs[0]
			input.ID = fmt.Sprintf("input-%d", len(document.Inputs))
			document.Inputs = append(document.Inputs, input)
		}
		draft, err := f.create(t, fmt.Sprintf("input-bound-%d", count), document)
		if count > auditprofile.MaxAuditProfileInputs {
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("oversized nested inputs accepted: %v", err)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, err = f.service.Start(f.ctx, f.start(draft)); err != nil {
			t.Fatalf("maximum nested input set failed: %v", err)
		}
	}
}

func TestEngagementMetadataAdmissionPrecedesPayloadReads(t *testing.T) {
	f := newEngagementFixture(t)
	profile, err := f.service.profiles.AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	oversized, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "oversized-engagement"}, artifacts.Payload{MediaType: pentestcontract.MediaType, Data: []byte(strings.Repeat(" ", pentestcontract.MaxDocumentBytes+1))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := f.store.Metadata(f.ctx, oversized.Ref)
	if err != nil {
		t.Fatal(err)
	}
	selection := DraftSelection{Inputs: map[string]contracts.ExactArtifact{"engagement": meta.Exact(), "checklist": f.checklist}}
	probe := &rulepackReadProbe{Repository: artifacts.NewPostgresRepository(f.pool), readErr: errors.New("unexpected engagement payload read")}
	if _, err = readAndVerifyInputs(f.ctx, artifacts.NewService(probe), f.project.ProjectID, profile, selection); !errors.Is(err, ErrInvalid) || probe.reads != 0 {
		t.Fatalf("oversized engagement materialized: reads=%d err=%v", probe.reads, err)
	}
	draft := storedRulepackDraft(t, f.ctx, f.service, profile, f.project.ProjectID, f.project.OwnerID, "stored-oversized-engagement", selection.Inputs)
	if _, err = f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized preflight=%v", err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized start=%v", err)
	}

	// A source-assisted document may classify a separate exact source while the
	// ordinary checklist selection still needs independent metadata validation.
	document := f.document("http://127.0.0.1")
	document.AccessMode = "source-assisted"
	document.Inputs[0].Role = "source"
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	written, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "metadata-engagement"}, artifacts.Payload{MediaType: pentestcontract.MediaType, Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := f.store.Metadata(f.ctx, written.Ref)
	if err != nil {
		t.Fatal(err)
	}
	badChecklist := f.checklist.Clone()
	badChecklist.Digest = serviceTestDigest("different-checklist")
	draft = storedRulepackDraft(t, f.ctx, f.service, profile, f.project.ProjectID, f.project.OwnerID, "stored-bad-checklist", map[string]contracts.ExactArtifact{"engagement": metadata.Exact(), "checklist": badChecklist})
	if _, err = f.service.Preflight(f.ctx, PreflightParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("preflight skipped ordinary exact input metadata: %v", err)
	}
}
