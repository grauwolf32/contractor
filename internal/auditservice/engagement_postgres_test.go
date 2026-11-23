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

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
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
		s = strings.Replace(s, "  inputs:\n", "  inputs:\n    engagement: {required: true, media_types: [application/vnd.contractor.engagement+json]}\n", 1)
		return strings.Replace(s, "active_operations: prohibited", "active_operations: allowed", 1)
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
	service, err := New(Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Profiles: &switchableProfileCatalog{snapshot: snapshot, available: true}, TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil }), CredentialGuard: engagementGuard{barrier}, Now: func() time.Time { return now }})
	if err != nil {
		t.Fatal(err)
	}
	cipher, err := credentials.NewTokenCipher([]byte(strings.Repeat("e", 32)))
	if err != nil {
		t.Fatal(err)
	}
	creds, err := credentials.NewRuntimeCredentialService(credentials.RuntimeCredentialServiceOptions{Pool: pool, Lease: leasetest.Hold(t, pool).Lease, Cipher: cipher, Usage: credentials.NewRuntimeCredentialRepository(pool), Projects: projectstore.NewPostgresStore(pool), Audits: hookstore.NewCredentialHolds(pool, artifactpgtest.NewStorage()), Barrier: barrier})
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "engagement-project", OwnerID: "engagement-owner", Kind: projectstore.KindProject, Name: "Engagement", IdempotencyKey: "engagement-project", RequestDigest: serviceTestDigest("engagement-project")})
	if err != nil {
		t.Fatal(err)
	}
	store, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
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
func (f *engagementFixture) document(origin string) engagementcontract.Document {
	return engagementcontract.Document{Schema: engagementcontract.Schema, Egress: engagementcontract.Egress{Routes: []engagementcontract.Route{{Origin: origin, PathPrefixes: []string{"/"}, Methods: []string{"GET"}, AddressRanges: []string{"127.0.0.0/8"}}}, Exclusions: []engagementcontract.Route{}}, Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{origin}}}}
}
func (f *engagementFixture) create(t *testing.T, id string, document engagementcontract.Document, extra ...map[string]contracts.ArtifactRef) (auditstore.Audit, error) {
	t.Helper()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: id}, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]contracts.ArtifactRef{"engagement": artifact.Ref, "checklist": f.checklist.Ref}
	for _, additional := range extra {
		for name, ref := range additional {
			inputs[name] = ref
		}
	}
	draft, _, err := f.service.CreateDraft(f.ctx, CreateDraftParams{AuditID: id, OwnerID: f.project.OwnerID, ProjectID: f.project.ProjectID, Profile: ProfileSelector{Name: "test-checklist", Version: "1"}, Inputs: inputs, RuntimeLabels: []string{}, Scope: Scope{Objective: "Check fixture"}, IdempotencyKey: id, RequestDigest: serviceTestDigest(id)})
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

func TestEngagementStartPinsAndReplay(t *testing.T) {
	f := newEngagementFixture(t)
	var requests atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests.Add(1) }))
	defer target.Close()
	document := f.document(target.URL)
	f.credential(t, "target-one", f.project.OwnerID)
	document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: []string{target.URL}, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "target-one", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-valid", document)
	if err != nil {
		t.Fatal(err)
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
	for _, bad := range []string{`null`, `{"schema":"contractor.audit.engagement-baseline.v1","artifact":null}`} {
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
	if started.Audit.Limits.MaxSubmittedRuns != 20 || started.Audit.Limits.MaxEvidenceBytes != 1048576 || started.Audit.DeadlineAt == nil || !started.Audit.DeadlineAt.Equal(f.now.Add(time.Hour)) {
		t.Fatalf("ordinary Audit limits not preserved: %+v", started.Audit)
	}
	if _, err := f.credentials.Delete(f.ctx, "target-one", f.project.OwnerID); !errors.Is(err, credentials.ErrRuntimeCredentialInUse) {
		t.Fatalf("held target delete=%v", err)
	}
	replay, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil || !replay.Replayed || string(replay.Audit.BaselineSnapshot) != string(started.Audit.BaselineSnapshot) {
		t.Fatalf("start replay changed authority: %v", err)
	}
	if requests.Load() != 0 || strings.Contains(string(started.Audit.BaselineSnapshot), "private-target-secret-canary") {
		t.Fatal("start executed target work or exposed secret")
	}
}

func TestEngagementAuditPinsOneAttemptAndHTTPLimits(t *testing.T) {
	f := newEngagementFixture(t)
	document := f.document("http://127.0.0.1")
	requests := uint32(7)
	document.Limits = &engagementcontract.HTTPLimits{RequestsPerSecond: &requests}
	draft, err := f.create(t, "single-attempt", document)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := auditprofile.DecodePinnedProgram(draft.ProfileSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Program.Limits.MaxAttempts <= 1 {
		t.Fatal("fixture must offer retries to prove the engagement override")
	}
	if draft.Limits.MaxItemRunAttempts != 1 {
		t.Fatalf("engagement retries=%d", draft.Limits.MaxItemRunAttempts)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil || baseline.Engagement.Document.Limits.RequestsPerSecond == nil || *baseline.Engagement.Document.Limits.RequestsPerSecond != requests {
		t.Fatalf("HTTP limits not pinned: %v", err)
	}
	decoded, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	clone := decoded.Engagement
	*clone.Document.Limits.RequestsPerSecond = 99
	if *baseline.Engagement.Document.Limits.RequestsPerSecond != requests {
		t.Fatal("cloned HTTP limits alias the baseline")
	}
}

func TestEngagementStartAcceptsSharedSitemapKnowledge(t *testing.T) {
	f := newEngagementFixtureWithEdits(t, func(worker string) string {
		return strings.Replace(worker, "  toolsets:\n", "  sitemap:\n    version: sitemap-access@1\n    collectObservations: false\n  toolsets:\n    - ref: sitemap@1\n      tools: [list_sitemap_entries]\n", 1)
	}, nil)
	draft, err := f.create(t, "shared-sitemap", f.document("http://127.0.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.Start(f.ctx, f.start(draft)); err != nil {
		t.Fatalf("shared sitemap prevents Audit start: %v", err)
	}
}

func TestEngagementStartValidatesOwnershipRevisionAndCredentials(t *testing.T) {
	f := newEngagementFixture(t)
	document := f.document("http://127.0.0.1")
	f.credential(t, "target-stale", f.project.OwnerID)
	document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "target-stale", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-stale", document)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		owner string
		rev   uint64
		want  error
	}{{"foreign", draft.Revision, auditstore.ErrNotFound}, {f.project.OwnerID, draft.Revision + 1, auditstore.ErrPrecondition}} {
		params := f.start(draft)
		params.OwnerID, params.ExpectedRevision = test.owner, test.rev
		if _, err := f.service.Start(f.ctx, params); !errors.Is(err, test.want) {
			t.Fatalf("owner/revision check=%v want %v", err, test.want)
		}
	}
	if _, err = f.credentials.Delete(f.ctx, "target-stale", f.project.OwnerID); err != nil {
		t.Fatal(err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("start accepted deleted credential: %v", err)
	}
	current, err := f.service.Get(f.ctx, f.project.OwnerID, draft.AuditID)
	if err != nil || current.State != auditstore.AuditDraft || current.BaselineSnapshot != nil {
		t.Fatalf("failed start left baseline: %+v %v", current, err)
	}
}

func TestEngagementOrdinaryInputsRetainExactRevisionsAndPurgeReleasesHolds(t *testing.T) {
	f := newEngagementFixtureWithEdits(t, nil, func(files map[string]string) {
		files["audit-profiles/checklist.yaml"] = strings.Replace(files["audit-profiles/checklist.yaml"], "  inputs:\n", "  inputs:\n    source: {required: false, media_types: [text/plain]}\n", 1)
	})
	document := f.document("http://127.0.0.1")
	source, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("original source snapshot")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	f.credential(t, "source-account", f.project.OwnerID)
	document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "source-account", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
	draft, err := f.create(t, "engagement-source", document, map[string]contracts.ArtifactRef{"source": source.Ref})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("replacement source")}, source.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	pinned := baseline.Inputs["source"]
	if !pinned.Ref.SameExact(source.Ref) {
		t.Fatal("ordinary input lost exact source revision")
	}
	held, err := f.store.ReadExact(f.ctx, pinned)
	if err != nil || string(held.Payload.Data) != "original source snapshot" {
		t.Fatalf("exact source replaced: %s %v", held.Payload.Data, err)
	}
	deleted, err := f.service.Delete(f.ctx, MutationParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "delete-source-audit", RequestDigest: serviceTestDigest("delete-source-audit")})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claim := claimAuditForPurge(t, f.ctx, store, draft.AuditID)
	if deleted.Audit.State == auditstore.AuditCancelling {
		if settled, err := store.SettleUndispatchedTasks(f.ctx, claim, auditstore.MaxReconcileRows); err != nil || settled != 0 {
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
				document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: document.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "race-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
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
		doc.Identities = append(doc.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: doc.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "foreign-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}})
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
		document func(*engagementcontract.Document)
		want     CompatibilityReason
	}{
		{"network-sandbox", func(s string) string { return strings.Replace(s, "local-workdir@1", "podman-network@1", 1) }, nil, ReasonEngagementCapabilityUnsupported},
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
			var before int
			if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_binding_revisions`).Scan(&before); err != nil {
				t.Fatal(err)
			}
			_, err = f.service.Start(f.ctx, f.start(draft))
			var unsupported *UnsupportedError
			if !errors.As(err, &unsupported) || !slices.Contains(unsupported.Reasons, test.want) {
				t.Fatalf("source/target closure reached materialization or lost reason %s: %v", test.want, err)
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

func TestEngagementStartValidatesProjectPolicyAndSuppressesDefaultIdentity(t *testing.T) {
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
	prohibited := contracts.ActiveChecksProhibited
	project, err = projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision, Name: project.Name, HTTPTarget: project.HTTPTarget, ActiveChecks: &prohibited})
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.service.Start(f.ctx, f.start(draft))
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) || !slices.Contains(unsupported.Reasons, ReasonEngagementActiveChecksUnsupported) {
		t.Fatalf("start ignored current Project policy or lost reason: %v", err)
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

func TestEngagementMetadataAdmissionPrecedesPayloadReads(t *testing.T) {
	f := newEngagementFixture(t)
	profile, err := f.service.profiles.AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	oversized, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "oversized-engagement"}, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: []byte(strings.Repeat(" ", engagementcontract.MaxDocumentBytes+1))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := f.store.Metadata(f.ctx, oversized.Ref)
	if err != nil {
		t.Fatal(err)
	}
	selection := DraftSelection{Inputs: map[string]contracts.ExactArtifact{"engagement": meta.Exact(), "checklist": f.checklist}}
	probe := &rulepackReadProbe{Repository: artifactpgtest.NewStorage().Repository(f.pool), readErr: errors.New("unexpected engagement payload read")}
	if _, err = readEngagement(f.ctx, artifacts.NewService(probe), f.project.ProjectID, profile, selection); !errors.Is(err, ErrInvalid) || probe.reads != 0 {
		t.Fatalf("oversized engagement materialized: reads=%d err=%v", probe.reads, err)
	}
	draft := storedRulepackDraft(t, f.ctx, f.service, profile, f.project.ProjectID, f.project.OwnerID, "stored-oversized-engagement", selection.Inputs)
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized start=%v", err)
	}

	// Ordinary input metadata is validated independently of engagement.
	document := f.document("http://127.0.0.1")
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	written, err := f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "metadata-engagement"}, artifacts.Payload{MediaType: engagementcontract.MediaType, Data: data}, nil)
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
	if _, err = f.service.Start(f.ctx, f.start(draft)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("start skipped ordinary exact input metadata: %v", err)
	}
}
