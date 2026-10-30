package credentials

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const targetCredentialOwner = "owner-target"

// targetCredentialMaterials builds one material of each target kind whose
// secret values carry canary.
func targetCredentialMaterials(t *testing.T, canary string) map[RuntimeCredentialKind]RuntimeCredentialMaterial {
	t.Helper()
	basic, err := NewTargetBasicCredential("alice", canary+"-password")
	if err != nil {
		t.Fatal(err)
	}
	bearer, err := NewTargetBearerCredential(canary + "-token")
	if err != nil {
		t.Fatal(err)
	}
	login, err := NewTargetCookiesCredential(map[string]string{"username": "alice", "password": canary + "-login"})
	if err != nil {
		t.Fatal(err)
	}
	materials := map[RuntimeCredentialKind]RuntimeCredentialMaterial{
		RuntimeCredentialTargetBasic: basic, RuntimeCredentialTargetBearer: bearer, RuntimeCredentialTargetCookies: login,
	}
	t.Cleanup(func() {
		for _, material := range materials {
			material.Destroy()
		}
	})
	return materials
}

func targetKindsInOrder() []RuntimeCredentialKind {
	return []RuntimeCredentialKind{RuntimeCredentialTargetBasic, RuntimeCredentialTargetBearer, RuntimeCredentialTargetCookies}
}

func newTargetCredentialService(t *testing.T, pool *pgxpool.Pool, audits RuntimeCredentialAuditHoldLookup) *RuntimeCredentialService {
	t.Helper()
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x76}, 32))
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Lease: leasetest.Hold(t, pool).Lease, Cipher: cipher,
		Usage: NewRuntimeCredentialRepository(pool), Projects: projectstore.NewPostgresStore(pool), Audits: audits,
		Barrier: NewLifecycleBarrier(),
	})
	if err != nil {
		t.Fatal(err)
	}
	return service
}

// resolveTargetCredential is the owner-scoped exact reference check an
// Audit start makes inside its own transaction while it holds the shared
// side of the credential lifecycle fence (RCF-R184).
func resolveTargetCredential(
	ctx context.Context, service *RuntimeCredentialService, pool *pgxpool.Pool,
	user RuntimeCredentialUser, ref runtimesettings.RuntimeCredentialRef,
) error {
	return service.WithCredentialReferences(ctx, func() error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			return NewRuntimeCredentialRepository(tx).ValidateRuntimeCredentialUse(
				ctx, user, ref.CredentialID, string(ref.Kind),
			)
		})
	})
}

// TestTargetCredentialPostgresCreateListResolveAndPrivateUse checks A1 of
// RCF-R184 on PostgreSQL: each target kind is created, listed and read as
// metadata without its secret; its exact reference resolves for its creator
// and Operations and is reported as missing for another owner, another kind,
// an unknown ID and after deletion; private use hands the consumer the
// canonical plaintext of the exact kind only; a deleted ID cannot be created
// again.
func TestTargetCredentialPostgresCreateListResolveAndPrivateUse(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	service := newTargetCredentialService(t, pool, staticAuditDispatchHolds{})
	const canary = "canary-target-a1-7d2e"
	materials := targetCredentialMaterials(t, canary)
	owner := RuntimeCredentialUser{UserID: targetCredentialOwner}
	for _, kind := range targetKindsInOrder() {
		id := "target-" + strings.TrimSuffix(strings.TrimPrefix(string(kind), "target-"), "@1")
		created, err := service.Create(ctx, RuntimeCredentialCreateRequest{
			CredentialID: id, Material: materials[kind], IdempotencyKey: "create-" + id, ActorID: owner.UserID,
		})
		if err != nil || created.Replayed || created.Credential.Kind != kind || created.Credential.CreatedBy != owner.UserID {
			t.Fatalf("create %s = (%+v, %v)", kind, created, err)
		}
	}
	page, err := service.List(ctx, "", 100)
	if err != nil || len(page) != 3 {
		t.Fatalf("list = (%+v, %v)", page, err)
	}
	encoded, err := json.Marshal(page)
	if err != nil || strings.Contains(string(encoded), canary) || strings.Contains(string(encoded), "alice") {
		t.Fatalf("listed metadata exposes material: %s, %v", encoded, err)
	}
	for _, metadata := range page {
		ref := runtimesettings.RuntimeCredentialRef{
			CredentialID: metadata.CredentialID, Kind: runtimesettings.RuntimeCredentialKind(metadata.Kind),
		}
		if err := ref.Validate(); err != nil {
			t.Fatalf("listed %s is not an exact reference: %v", metadata.CredentialID, err)
		}
		read, err := service.Get(ctx, metadata.CredentialID)
		if err != nil || read != metadata {
			t.Fatalf("get %s = (%+v, %v), want %+v", metadata.CredentialID, read, err, metadata)
		}
		for name, user := range map[string]RuntimeCredentialUser{
			"creator": owner, "Operations": {UserID: "operator", Operations: true},
		} {
			if err := resolveTargetCredential(ctx, service, pool, user, ref); err != nil {
				t.Fatalf("%s resolves %s: %v", name, ref.CredentialID, err)
			}
		}
		wrongKind := ref
		wrongKind.Kind = runtimesettings.RuntimeCredentialTargetCookies
		if ref.Kind == runtimesettings.RuntimeCredentialTargetCookies {
			wrongKind.Kind = runtimesettings.RuntimeCredentialTargetBasic
		}
		for name, check := range map[string]struct {
			user RuntimeCredentialUser
			ref  runtimesettings.RuntimeCredentialRef
		}{
			"another owner": {RuntimeCredentialUser{UserID: "owner-other"}, ref},
			"another kind":  {owner, wrongKind},
			"an unknown ID": {owner, runtimesettings.RuntimeCredentialRef{CredentialID: "target-missing", Kind: ref.Kind}},
		} {
			if err := resolveTargetCredential(ctx, service, pool, check.user, check.ref); !errors.Is(err, ErrRuntimeCredentialNotFound) {
				t.Fatalf("%s resolves %s = %v, want not found", name, ref.CredentialID, err)
			}
		}

		var seen []byte
		err = service.UsePlaintext(ctx, ref.CredentialID, []runtimesettings.RuntimeCredentialKind{ref.Kind},
			func(kind runtimesettings.RuntimeCredentialKind, plaintext []byte) error {
				if kind != ref.Kind {
					return fmt.Errorf("private use of %s returned kind %s", ref.CredentialID, kind)
				}
				material := materials[RuntimeCredentialKind(kind)]
				return material.WithPlaintext(func(_ RuntimeCredentialKind, canonical []byte) error {
					if !bytes.Equal(plaintext, canonical) {
						return errors.New("private use returned other plaintext")
					}
					seen = plaintext
					return nil
				})
			})
		if err != nil {
			t.Fatalf("private use of %s: %v", ref.CredentialID, err)
		}
		if len(seen) == 0 || slices.ContainsFunc(seen, func(value byte) bool { return value != 0 }) {
			t.Fatalf("private use of %s left its plaintext readable", ref.CredentialID)
		}
		if err := service.UsePlaintext(ctx, ref.CredentialID, []runtimesettings.RuntimeCredentialKind{wrongKind.Kind},
			func(runtimesettings.RuntimeCredentialKind, []byte) error {
				return errors.New("a wrong-kind consumer was called")
			}); !errors.Is(err, ErrRuntimeCredentialNotFound) {
			t.Fatalf("private use of %s as %s = %v, want not found", ref.CredentialID, wrongKind.Kind, err)
		}

		if deleted, err := service.Delete(ctx, ref.CredentialID, owner.UserID); err != nil || deleted.Replayed {
			t.Fatalf("delete %s = (%+v, %v)", ref.CredentialID, deleted, err)
		}
		if err := resolveTargetCredential(ctx, service, pool, owner, ref); !errors.Is(err, ErrRuntimeCredentialNotFound) {
			t.Fatalf("deleted %s resolves = %v, want not found", ref.CredentialID, err)
		}
		if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
			CredentialID: ref.CredentialID, Material: materials[RuntimeCredentialKind(ref.Kind)],
			IdempotencyKey: "recreate-" + ref.CredentialID, ActorID: owner.UserID,
		}); !errors.Is(err, ErrRuntimeCredentialConflict) {
			t.Fatalf("re-create of deleted %s = %v, want a conflict", ref.CredentialID, err)
		}
	}
}

// TestTargetCredentialPostgresLostResponseReplay checks A4 of RCF-R184: a
// create or delete whose answer was lost and is retried with its original
// key returns the original identity or tombstone without another revision
// or changed ciphertext; the key with other material conflicts, and another
// key cannot take the identity.
func TestTargetCredentialPostgresLostResponseReplay(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	service := newTargetCredentialService(t, pool, staticAuditDispatchHolds{})
	materials := targetCredentialMaterials(t, "canary-target-a4-91b0")
	changed := targetCredentialMaterials(t, "canary-target-a4-other")
	repository := NewRuntimeCredentialRepository(pool)
	for _, kind := range targetKindsInOrder() {
		id := "replay-" + strings.TrimSuffix(strings.TrimPrefix(string(kind), "target-"), "@1")
		request := RuntimeCredentialCreateRequest{
			CredentialID: id, Material: materials[kind], IdempotencyKey: "create-" + id, ActorID: targetCredentialOwner,
		}
		first, err := service.Create(ctx, request)
		if err != nil || first.Replayed {
			t.Fatalf("create %s = (%+v, %v)", id, first, err)
		}
		stored, err := repository.GetAnyRecord(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		replayed, err := service.Create(ctx, request)
		if err != nil || !replayed.Replayed || !sameRuntimeCredentialMetadata(replayed.Credential, first.Credential) {
			t.Fatalf("retried create %s = (%+v, %v), want a replay of %+v", id, replayed, err, first.Credential)
		}
		otherContent := request
		otherContent.Material = changed[kind]
		if _, err := service.Create(ctx, otherContent); !errors.Is(err, contracts.ErrIdempotencyConflict) {
			t.Fatalf("create key %s with other material = %v, want an idempotency conflict", id, err)
		}
		otherKey := request
		otherKey.IdempotencyKey = "another-" + id
		if _, err := service.Create(ctx, otherKey); !errors.Is(err, ErrRuntimeCredentialConflict) {
			t.Fatalf("another key for %s = %v, want a conflict", id, err)
		}
		after, err := repository.GetAnyRecord(ctx, id)
		if err != nil || after.Metadata != stored.Metadata || !bytes.Equal(after.Envelope.Nonce, stored.Envelope.Nonce) ||
			!bytes.Equal(after.Envelope.Ciphertext, stored.Envelope.Ciphertext) {
			t.Fatalf("retries changed the stored revision of %s: %v", id, err)
		}
		var rows int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM runtime_credentials WHERE credential_id = $1`, id).Scan(&rows); err != nil || rows != 1 {
			t.Fatalf("rows of %s = (%d, %v)", id, rows, err)
		}

		if deleted, err := service.Delete(ctx, id, targetCredentialOwner); err != nil || deleted.Replayed {
			t.Fatalf("delete %s = (%+v, %v)", id, deleted, err)
		}
		tombstone, err := repository.GetTombstone(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		if deleted, err := service.Delete(ctx, id, targetCredentialOwner); err != nil || !deleted.Replayed {
			t.Fatalf("retried delete %s = (%+v, %v), want a replay", id, deleted, err)
		}
		if again, err := repository.GetTombstone(ctx, id); err != nil || again != tombstone {
			t.Fatalf("retried delete changed the tombstone of %s: (%+v, %v), want %+v", id, again, err, tombstone)
		}
		if _, err := service.Get(ctx, id); !errors.Is(err, ErrRuntimeCredentialNotFound) {
			t.Fatalf("deleted %s is readable: %v", id, err)
		}
	}
}

// TestTargetCredentialPostgresUseWipesMaterialOnFailureAndCancellation checks
// A5 of RCF-R184: when the private consumer fails or its context is
// cancelled, use returns a fixed error or the context error that repeats no
// secret, and the plaintext it handed out is wiped.
func TestTargetCredentialPostgresUseWipesMaterialOnFailureAndCancellation(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	service := newTargetCredentialService(t, pool, staticAuditDispatchHolds{})
	const canary = "canary-target-a5-3f8c"
	materials := targetCredentialMaterials(t, canary)
	if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "wiped-login", Material: materials[RuntimeCredentialTargetCookies],
		IdempotencyKey: "create-wiped-login", ActorID: targetCredentialOwner,
	}); err != nil {
		t.Fatal(err)
	}
	kinds := []runtimesettings.RuntimeCredentialKind{runtimesettings.RuntimeCredentialTargetCookies}
	for _, test := range []struct {
		name    string
		consume func(context.CancelFunc) error
		want    func(error) bool
	}{
		{"consumer failure", func(context.CancelFunc) error {
			return errors.New("login failed with " + canary)
		}, func(err error) bool { return err != nil && !errors.Is(err, context.Canceled) }},
		{"cancellation", func(cancel context.CancelFunc) error {
			cancel()
			return errors.New("cancelled with " + canary)
		}, func(err error) bool { return errors.Is(err, context.Canceled) }},
	} {
		useContext, cancelUse := context.WithCancel(ctx)
		var seen []byte
		err := service.UsePlaintext(useContext, "wiped-login", kinds,
			func(_ runtimesettings.RuntimeCredentialKind, plaintext []byte) error {
				seen = plaintext
				if !bytes.Contains(plaintext, []byte(canary)) {
					return errors.New("the consumer did not receive the material")
				}
				return test.consume(cancelUse)
			})
		cancelUse()
		if !test.want(err) || strings.Contains(fmt.Sprint(err), canary) {
			t.Fatalf("%s: use error = %v", test.name, err)
		}
		if len(seen) == 0 || slices.ContainsFunc(seen, func(value byte) bool { return value != 0 }) {
			t.Fatalf("%s: the plaintext handed to the consumer was not wiped", test.name)
		}
	}
}

// TestTargetCredentialDeleteRacesAuditHoldWithoutDeadlock checks A3 of
// RCF-R184 on PostgreSQL with one and with four connections: an Audit start
// that resolves an exact target reference and commits a held baseline naming
// it races deletion of that credential. Either the hold commits first and
// deletion is refused with the Audit as its user, or deletion commits first
// and the hold finds no credential and holds nothing; neither waits for a
// second pool connection.
func TestTargetCredentialDeleteRacesAuditHoldWithoutDeadlock(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, maxConns := range []int32{1, 4} {
		t.Run(fmt.Sprintf("MaxConns=%d", maxConns), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			isolated := isolatedCredentialPool(t, ctx, databaseURL)
			configuration := isolated.Config().Copy()
			configuration.MaxConns, configuration.MinConns, configuration.MinIdleConns = maxConns, 0, 0
			pool, err := pgxpool.NewWithConfig(ctx, configuration)
			if err != nil {
				t.Fatal(err)
			}
			defer pool.Close()
			audits := auditstoretest.NewStore(pool)
			service := newTargetCredentialService(t, pool, audits)
			materials := targetCredentialMaterials(t, "canary-target-a3-c51d")
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-target-race", OwnerID: targetCredentialOwner, Kind: projectstore.KindProject,
				Name: "Target race", IdempotencyKey: "project-target-race", RequestDigest: raceDigest("1"),
			})
			if err != nil {
				t.Fatal(err)
			}
			owner := RuntimeCredentialUser{UserID: project.OwnerID}
			round := 0
			race := func(kind RuntimeCredentialKind, order string) (holdErr, deleteErr error, auditID string) {
				t.Helper()
				round++
				id := fmt.Sprintf("race-%d", round)
				auditID = fmt.Sprintf("audit-race-%d", round)
				if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
					CredentialID: id, Material: materials[kind], IdempotencyKey: "create-" + id, ActorID: owner.UserID,
				}); err != nil {
					t.Fatal(err)
				}
				audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
					AuditID: auditID, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
					Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: raceDigest("2")},
					ProfileSnapshot: json.RawMessage(`{"name":"profile"}`), InputSelection: json.RawMessage(`{"inputs":{}}`),
					Limits: auditstore.Limits{
						MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
						MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
					},
					IdempotencyKey: "draft-" + auditID, RequestDigest: raceDigest("3"),
				})
				if err != nil {
					t.Fatal(err)
				}
				ref := runtimesettings.RuntimeCredentialRef{CredentialID: id, Kind: runtimesettings.RuntimeCredentialKind(kind)}
				hold := func() error {
					return service.WithCredentialReferences(ctx, func() error {
						return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
							if err := NewRuntimeCredentialRepository(tx).ValidateRuntimeCredentialUse(
								ctx, owner, ref.CredentialID, string(ref.Kind),
							); err != nil {
								return err
							}
							revision := "r1"
							_, _, err := auditstoretest.NewTxStore(tx).MaterializeRound(ctx, auditstore.MaterializeRoundParams{
								OwnerID: project.OwnerID, AuditID: auditID, ExpectedRevision: audit.Revision,
								RoundID: "round-" + auditID, RoundOrdinal: 1,
								Manifest: auditstore.ExactArtifact{
									Ref:    contracts.ArtifactRef{Namespace: "audits", Name: "round-" + auditID, Revision: &revision},
									Digest: raceDigest("a"), MediaType: "application/zip", SizeBytes: 1,
								},
								BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[],"llmCredentialIds":[],` +
									`"runtimeCredentialIds":["` + id + `"]}`),
								DeadlineAt: time.Now().Add(time.Hour), Items: []auditstore.MaterializedItem{},
								IdempotencyKey: "start-" + auditID, RequestDigest: raceDigest("4"),
							})
							return err
						})
					})
				}
				remove := func() error {
					_, err := service.Delete(ctx, id, owner.UserID)
					return err
				}
				switch order {
				case "hold first":
					holdErr = hold()
					deleteErr = remove()
				case "delete first":
					deleteErr = remove()
					holdErr = hold()
				default:
					start := make(chan struct{})
					var wait sync.WaitGroup
					wait.Add(2)
					go func() { defer wait.Done(); <-start; holdErr = hold() }()
					go func() { defer wait.Done(); <-start; deleteErr = remove() }()
					close(start)
					wait.Wait()
				}
				var inUse *RuntimeCredentialInUseError
				held, listErr := audits.ListHeldAuditIDsByRuntimeCredential(ctx, id, 10)
				_, getErr := service.Get(ctx, id)
				switch {
				case listErr != nil:
					t.Fatalf("%s %s: list holds: %v", kind, order, listErr)
				case holdErr == nil && errors.As(deleteErr, &inUse) &&
					slices.Equal(inUse.Usage.AuditIDs, []string{auditID}) &&
					slices.Equal(held, []string{auditID}) && getErr == nil:
				case errors.Is(holdErr, ErrRuntimeCredentialNotFound) && deleteErr == nil &&
					len(held) == 0 && errors.Is(getErr, ErrRuntimeCredentialNotFound):
				default:
					t.Fatalf("%s %s: hold = %v, delete = %v, held = %v, credential = %v",
						kind, order, holdErr, deleteErr, held, getErr)
				}
				return holdErr, deleteErr, auditID
			}
			for _, kind := range targetKindsInOrder() {
				if holdErr, _, _ := race(kind, "hold first"); holdErr != nil {
					t.Fatalf("%s: hold first failed: %v", kind, holdErr)
				}
				if _, deleteErr, _ := race(kind, "delete first"); deleteErr != nil {
					t.Fatalf("%s: delete first failed: %v", kind, deleteErr)
				}
				for range 4 {
					race(kind, "concurrently")
				}
			}
		})
	}
}

func sameRuntimeCredentialMetadata(left, right RuntimeCredentialMetadata) bool {
	return left.CredentialID == right.CredentialID && left.Kind == right.Kind &&
		left.CreatedBy == right.CreatedBy && left.CreatedAt.Equal(right.CreatedAt)
}

func raceDigest(character string) string { return "sha256:" + strings.Repeat(character, 64) }
