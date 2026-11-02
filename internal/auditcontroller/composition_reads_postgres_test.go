//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// detailHydratesPreparation selects every started Audit for a hydrated
// detail, as the public detail does for an Audit that pins a prepare role;
// the Audits of these tests pin one.
func detailHydratesPreparation(audit auditstore.Audit) bool {
	return audit.Phase != auditdomain.AuditPhaseNotStarted
}

func TestPostgresCompositionReadsOnlyCommittedInventoryAndRetainedOutputs(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	claim, snapshot := initialInventoryClaim(t, ctx, h)
	owner, id := snapshot.Audit.OwnerID, snapshot.Audit.AuditID
	params, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
	if err != nil || reason != nil {
		t.Fatalf("prepare: %v %v", reason, err)
	}
	if staged, err := h.service.Detail(ctx, owner, snapshot.Audit, detailHydratesPreparation); err != nil ||
		staged.GeneratedInventory != nil || staged.Preparation == nil || staged.Audit.Phase != auditdomain.AuditPhaseInventory {
		t.Fatalf("staging became inventory: %+v %v", staged, err)
	}
	ref := params.Inventory.Artifact.Ref
	if _, err := h.service.ReadArtifact(ctx, owner, id, auditstore.InitialInventoryLogicalKey, *ref.Revision); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("staging became readable: %v", err)
	}
	if _, _, err := h.audits.AcceptInitialRound(ctx, params); err != nil {
		t.Fatal(err)
	}
	audit, err := h.audits.Get(ctx, owner, id)
	if err != nil {
		t.Fatal(err)
	}
	accepted, err := h.service.Detail(ctx, owner, audit, detailHydratesPreparation)
	view := accepted.GeneratedInventory
	if err != nil || accepted.Audit.Revision != audit.Revision || view == nil || view.ItemCount != len(params.Items) ||
		!view.Snapshot.Ref.SameExact(ref) || !view.Worklist.Ref.SameExact(params.Manifest.Ref) {
		t.Fatalf("accepted inventory: %+v %v", view, err)
	}
	// The detail of an earlier state is the Audit as of a later commit, never
	// a conflict (ASC-R149).
	if later, err := h.service.Detail(ctx, owner, snapshot.Audit, detailHydratesPreparation); err != nil ||
		later.Audit.Revision != audit.Revision || later.Audit.Phase != auditdomain.AuditPhaseRounds ||
		later.GeneratedInventory == nil || !later.GeneratedInventory.Snapshot.Ref.SameExact(ref) {
		t.Fatalf("detail of an earlier state: %+v %v", later, err)
	}
	if _, err := h.service.Detail(ctx, "foreign-owner", audit, detailHydratesPreparation); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign inventory: %v", err)
	}
	prepared := preparationExecutions(t, ctx, h, 1)[0]
	output, err := h.audits.GetPreparationOutput(ctx, id, "z-seed", "list")
	if err != nil {
		t.Fatal(err)
	}
	key, revision := "prepare:"+prepared.ExecutionID+":list", *output.Output.Retained.Ref.Revision
	if _, err := h.service.ReadArtifact(ctx, owner, id, key, revision); err != nil {
		t.Fatalf("accepted output: %v", err)
	}
	for _, request := range [][3]string{{"foreign-owner", key, revision}, {owner, "uncommitted", revision}, {owner, key, "other-revision"}} {
		if _, err := h.service.ReadArtifact(ctx, request[0], id, request[1], request[2]); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("unauthorized or mutable lookup: %v", err)
		}
	}
	if err := rundeletion.New(h.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, owner, *prepared.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.ReadArtifact(ctx, owner, id, key, revision); err != nil {
		t.Fatalf("source Run deletion removed retained output: %v", err)
	}
	audit, _ = h.audits.Get(ctx, owner, id)
	if retained, err := h.service.Detail(ctx, owner, audit, detailHydratesPreparation); err != nil || retained.GeneratedInventory == nil ||
		retained.GeneratedInventory.Sources["source"].Digest != output.Output.Retained.Digest {
		t.Fatalf("source deletion changed inventory provenance: %+v %v", retained, err)
	}
}

// statementInterleave records the statements of a traced pool and runs
// interleave once, before the first of them whose text contains marker.
type statementInterleave struct {
	mu         sync.Mutex
	marker     string
	interleave func() error
	ran        bool
	err        error
	statements []string
}

func (s *statementInterleave) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	s.mu.Lock()
	s.statements = append(s.statements, data.SQL)
	interleave := s.interleave
	if interleave == nil || !strings.Contains(data.SQL, s.marker) {
		s.mu.Unlock()
		return ctx
	}
	s.interleave = nil
	s.mu.Unlock()
	err := interleave()
	s.mu.Lock()
	s.ran, s.err = true, err
	s.mu.Unlock()
	return ctx
}

func (*statementInterleave) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// arm runs interleave before the next statement that contains marker and
// forgets the statements recorded so far.
func (s *statementInterleave) arm(marker string, interleave func() error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.marker, s.interleave, s.ran, s.err, s.statements = marker, interleave, false, nil, nil
}

// recorded returns the statements recorded since arm.
func (s *statementInterleave) recorded() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.statements...)
}

// TestPostgresAuditDetailReadsGeneratedInventoryFromOneCommit commits an
// owner pause between the statements of the detail of a rounds-phase Audit
// with generated inventory, and checks that the detail and a start response
// carry the Audit, its preparation, its inventory and its current Round as
// of one commit (V414-018).
func TestPostgresAuditDetailReadsGeneratedInventoryFromOneCommit(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	claim, snapshot := initialInventoryClaim(t, ctx, h)
	owner, id := snapshot.Audit.OwnerID, snapshot.Audit.AuditID
	params, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
	if err != nil || reason != nil {
		t.Fatalf("prepare: %v %v", reason, err)
	}
	if _, _, err := h.audits.AcceptInitialRound(ctx, params); err != nil {
		t.Fatal(err)
	}
	active, err := h.audits.Get(ctx, owner, id)
	if err != nil || active.Phase != auditdomain.AuditPhaseRounds || active.CurrentRoundID == nil {
		t.Fatalf("accepted Audit: %+v %v", active, err)
	}
	trace := &statementInterleave{}
	tracedConfig := h.pool.Config()
	tracedConfig.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, tracedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	service, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(),
		Pool: traced, Profiles: h.snapshot, TransactionLLMCredentials: controllerTransactionCredentialLookup(),
		CredentialGuard: controllerCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	pause := func() error {
		_, _, err := h.audits.Transition(ctx, auditstore.TransitionParams{OwnerID: owner, AuditID: id,
			ExpectedRevision: active.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused,
			IdempotencyKey: "pause-during-detail", RequestDigest: postgresDigest("pause-during-detail")})
		return err
	}
	// The pause commits after the snapshot's Audit and preparation statements
	// and before its initial-inventory link statement.
	trace.arm("FROM audit_artifact_links", pause)
	detail, err := service.Detail(ctx, owner, active, detailHydratesPreparation)
	if !trace.ran || trace.err != nil {
		t.Fatalf("pause between the detail statements ran=%t err=%v", trace.ran, trace.err)
	}
	if err != nil || detail.Audit.Revision != active.Revision || detail.Audit.State != auditstore.AuditActive ||
		detail.Preparation["z-seed"].Status != auditdomain.PreparationAccepted || detail.GeneratedInventory == nil ||
		detail.GeneratedInventory.ItemCount != len(params.Items) {
		t.Fatalf("detail across a pause = %+v %v", detail, err)
	}
	// BEGIN, the Audit, its preparation, the link, Round 1 and COMMIT; the
	// exact snapshot is read after the commit.
	statements := trace.recorded()
	commit := -1
	for index, statement := range statements {
		if strings.EqualFold(strings.TrimSpace(statement), "commit") {
			commit = index
			break
		}
	}
	if commit != 5 || !strings.Contains(strings.ToLower(statements[0]), "repeatable read read only") {
		t.Fatalf("detail snapshot statements = %q", statements)
	}
	paused, err := h.audits.Get(ctx, owner, id)
	if err != nil || paused.State != auditstore.AuditPaused || paused.Revision <= active.Revision {
		t.Fatalf("paused Audit: %+v %v", paused, err)
	}
	// A later detail shows the pause with its own revision and the same
	// inventory.
	later, err := service.Detail(ctx, owner, active, detailHydratesPreparation)
	if err != nil || later.Audit.Revision != paused.Revision || later.Audit.State != auditstore.AuditPaused ||
		later.GeneratedInventory == nil || !later.GeneratedInventory.Snapshot.Ref.SameExact(detail.GeneratedInventory.Snapshot.Ref) {
		t.Fatalf("detail after the pause = %+v %v", later, err)
	}
	// A start response read once the Audit entered its first Round carries
	// that Round and only its items, from the commit of its Audit
	// (AUD-R226), although the start was read in the inventory phase.
	started, err := service.StartedDetail(ctx, owner,
		auditservice.StartedAudit{Audit: snapshot.Audit, Items: []auditstore.Item{}, Replayed: true}, detailHydratesPreparation)
	if err != nil || started.Audit.Revision != paused.Revision || started.Round == nil ||
		started.Round.RoundID != *paused.CurrentRoundID || len(started.Items) != len(params.Items) || started.GeneratedInventory == nil {
		t.Fatalf("start response after the first Round = %+v %v", started, err)
	}
	for _, item := range started.Items {
		if item.RoundID != started.Round.RoundID {
			t.Fatalf("start response item of another Round: %+v", item)
		}
	}
}
