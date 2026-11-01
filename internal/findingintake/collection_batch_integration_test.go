//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type collectionQueryCounter struct {
	count       atomic.Int64
	runReads    atomic.Int64
	directReads atomic.Int64
}

func (c *collectionQueryCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	c.count.Add(1)
	if strings.Contains(data.SQL, "FROM workflow_runs WHERE run_id = $1") {
		c.runReads.Add(1)
	}
	if strings.Contains(data.SQL, "blob.payload") && len(data.Args) >= 4 &&
		data.Args[2] == "outputs" && data.Args[3] == "result" {
		c.directReads.Add(1)
	}
	return ctx
}
func (*collectionQueryCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestPostgresRunProposalPagesUseBatchedReadsAndSkipRetainedDocuments(t *testing.T) {
	f := newDeletionImportFixture(t)
	// The test counts queries, not time; go test -timeout bounds it.
	ctx := t.Context()
	f.ctx = ctx
	for i := 1; i < 200; i++ {
		insertAuditChildReceipt(t, f, fmt.Sprintf("candidate-%03d", i))
	}
	trace := &collectionQueryCounter{}
	actor, _ := deletionActorPool(t, f, trace)
	reader, err := New(actor)
	if err != nil {
		t.Fatal(err)
	}
	trace.count.Store(0)
	page, err := reader.ListRun(ctx, f.request.OwnerID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(page) != 200 {
		t.Fatalf("Run page: %d receipts, %v", len(page), err)
	}
	if got := trace.count.Load(); got > 9 { // page + holds + seven artifact batches
		t.Fatalf("200-receipt Run page used %d queries, want at most 9", got)
	}
	if page[0].Document.ClientKey == "" || page[199].Document.ClientKey == "" {
		t.Fatal("batched Run page omitted proposal documents")
	}
	start := time.Now()
	requests := make([]ImportRequest, len(page))
	for index, receipt := range page {
		requests[index] = f.request
		requests[index].Proposal = receipt.Proposal.Ref
	}
	if err := f.intake.RetainAuditCollectionBatch(ctx, requests); err != nil {
		t.Fatal(err)
	}
	t.Logf("retained 200 proposals in %s", time.Since(start))
	trace.count.Store(0)
	collected, err := reader.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(collected) != 200 {
		t.Fatalf("collection page: %d receipts, %v", len(collected), err)
	}
	if got := trace.count.Load(); got != 2 {
		t.Fatalf("retained collection page used %d queries, want 2", got)
	}
	for _, receipt := range collected {
		if !receipt.Retained || receipt.Receipt.Document.ClientKey != "" {
			t.Fatal("retained collection page rehydrated a proposal")
		}
	}
}

const largeDirectProposals = 2000

func newLargeDirectCollectionFixture(t *testing.T) deletionImportFixture {
	t.Helper()
	f := newDirectCollectionFixture(t)
	// Seeding 2,000 proposals takes as long as the machine needs; go test
	// -timeout bounds it.
	ctx := t.Context()
	f.ctx = ctx
	verifications := make([]auditdomain.DirectVerificationResult, largeDirectProposals)
	for i := range verifications {
		key := fmt.Sprintf("candidate-%04d", i)
		insertAuditChildReceipt(t, f, key)
		verifications[i] = auditdomain.DirectVerificationResult{
			InvocationID: key + "-invocation", ClientKey: key,
			Assessment: "supported", Summary: "Verified exact candidate.", EvidenceIDs: []string{},
		}
	}
	encoded, err := json.Marshal(auditdomain.DirectVerificationSet{
		Schema: auditdomain.DirectVerificationsSchema, Verifications: verifications,
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	runArtifacts, err := artifactService.Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Write(ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "direct-result"},
		artifacts.Payload{MediaType: auditdomain.DirectVerificationsMediaType, Data: encoded}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, artifactService, f.request.RunID, "result", output.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, artifactService, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(f.pool).TransitionRun(ctx, f.request.RunID,
		runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "collection_fixture_succeeded"}); err != nil {
		t.Fatal(err)
	}
	return f
}

// Every attempt ends once it has done the work of one bounded transaction,
// the least the production collection timeout allows, and must keep what it
// committed, so the collection finishes across attempts however long the
// whole Run takes.
func TestPostgresLargeDirectCollectionFinishesAcrossShortAttempts(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	trace := &collectionQueryCounter{}
	transactions := &collectionTransactionTracer{}
	actorPool, _ := deletionActorPool(t, f, multiQueryTracer{trace, transactions})
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	start := time.Now()
	attempts := collectAcrossAttempts(t, f, transactions, true)
	t.Logf("fresh 2,001-proposal collection completed in %d attempt(s), %s", attempts, time.Since(start))
	assertFixtureHolds(t, f, largeDirectProposals+1)
	// Each attempt resolves the terminal Run and its direct output once.
	if trace.runReads.Load() != int64(attempts) || trace.directReads.Load() != int64(attempts) {
		t.Fatalf("collection resolved Run/output %d/%d times over %d attempt(s)",
			trace.runReads.Load(), trace.directReads.Load(), attempts)
	}
	transactions.assertBounded(t)
}

// An attempt that stops after committing part of the Run keeps that part,
// and the next attempt retains only the rest. The first attempt is ended
// right after its first retention transaction commits, as an expiring
// deadline would, independent of collection speed.
func TestPostgresLargeDirectCollectionFinishesAfterTimedOutAttempt(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	ctx := f.ctx
	const proposals = largeDirectProposals
	firstCtx, firstCancel := context.WithCancel(ctx)
	defer firstCancel()
	interrupted := &collectionTransactionTracer{interrupt: firstCancel}
	actorPool, _ := deletionActorPool(t, f, interrupted)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	first := f
	first.intake = actor
	if err := collectFixtureProposals(firstCtx, first, true); err == nil {
		t.Fatal("interrupted first collection unexpectedly finished")
	} else {
		t.Logf("first collection stopped: %v", err)
	}
	firstCount := countFixtureHolds(t, f)
	if interrupted.committed != 1 || firstCount == 0 || firstCount >= proposals+1 {
		t.Fatalf("interrupted collection committed %d transaction(s) retaining %d/%d proposals",
			interrupted.committed, firstCount, proposals+1)
	}
	retried := &collectionTransactionTracer{}
	retryPool, _ := deletionActorPool(t, f, retried)
	retryActor, err := New(retryPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = retryActor
	attempts := collectAcrossAttempts(t, f, retried, true)
	t.Logf("collected %d remaining proposals in %d attempt(s)", proposals+1-firstCount, attempts)
	assertFixtureHolds(t, f, proposals+1)
	retried.assertBounded(t)
	trace := &collectionQueryCounter{}
	actorPool, _ = deletionActorPool(t, f, trace)
	actor, err = New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	trace.count.Store(0)
	if err := collectFixtureProposals(ctx, f, true); err != nil {
		t.Fatal(err)
	}
	if got := trace.count.Load(); got > 2*((proposals+1)/200+1) || trace.directReads.Load() != 0 {
		t.Fatalf("completed collection replay used %d queries and %d output reads", got, trace.directReads.Load())
	}
}

// A page of proposals that each cite many evidence revisions may not commit
// in one transaction within the production timeout. Bounded transactions
// keep each attempt's progress, so the page completes across attempts. The
// first attempt is ended after its first retention transaction commits.
func TestPostgresEvidenceHeavyCollectionPageCommitsBoundedTransactions(t *testing.T) {
	const proposals, evidencePerProposal = 200, 48
	f := newDeletionImportFixture(t)
	// Attempts end through the transaction tracer, not a deadline; go test
	// -timeout bounds the test.
	ctx := t.Context()
	f.ctx = ctx
	evidence := writeFixtureEvidence(t, f, evidencePerProposal)
	for i := range proposals {
		insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("evidence-candidate-%03d", i), evidence)
	}
	firstCtx, firstCancel := context.WithCancel(ctx)
	defer firstCancel()
	transactions := &collectionTransactionTracer{interrupt: firstCancel}
	actorPool, _ := deletionActorPool(t, f, transactions)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	if err := collectFixtureProposals(firstCtx, f, false); err == nil {
		t.Fatal("interrupted first collection unexpectedly finished")
	}
	if firstCount := countFixtureHolds(t, f); firstCount == 0 || firstCount >= proposals {
		t.Fatalf("interrupted attempt retained %d/%d proposals", firstCount, proposals+1)
	}
	start := time.Now()
	attempts := collectAcrossAttempts(t, f, transactions, false)
	t.Logf("retained %d proposals with %d evidence revisions each in %d more attempt(s), %d transactions, %s",
		proposals, evidencePerProposal, attempts, transactions.committed, time.Since(start))
	assertFixtureHolds(t, f, proposals+1)
	var retainedEvidence int
	if err := f.pool.QueryRow(ctx, `
SELECT COALESCE(sum(jsonb_array_length(evidence)), 0)::integer
  FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&retainedEvidence); err != nil ||
		retainedEvidence != proposals*evidencePerProposal {
		t.Fatalf("retained evidence = %d, %v; want %d", retainedEvidence, err, proposals*evidencePerProposal)
	}
	if transactions.committed < proposals*(1+evidencePerProposal)/auditdomain.MaximumCollectionTransactionWork {
		t.Fatalf("page committed in %d transaction(s)", transactions.committed)
	}
	transactions.assertBounded(t)
}

// collectAcrossAttempts repeats collection as the Controller does after an
// attempt's operation timeout expires, until one attempt finishes. The
// production timeout promises an attempt time for one bounded transaction
// (RetainAuditCollectionBatch), not more, so transactions, the tracer of
// f.intake's pool, ends each attempt at its first artifact import beyond
// that work, however fast or loaded the machine is. A transaction larger
// than the bound therefore rolls back, and every attempt that stops must
// have retained new proposals; holds never exceed the fixture's proposals,
// so the loop ends. It returns the number of attempts.
func collectAcrossAttempts(
	t *testing.T, f deletionImportFixture, transactions *collectionTransactionTracer, sourceSucceeded bool,
) int {
	t.Helper()
	for attempt := 1; ; attempt++ {
		before := countFixtureHolds(t, f)
		attemptCtx, cancel := context.WithCancel(f.ctx)
		transactions.limitAttempt(cancel)
		err := collectFixtureProposals(attemptCtx, f, sourceSucceeded)
		interrupted := attemptCtx.Err() != nil
		transactions.limitAttempt(nil)
		cancel()
		if err == nil {
			return attempt
		}
		if !interrupted {
			t.Fatalf("collection attempt %d: %v", attempt, err)
		}
		if after := countFixtureHolds(t, f); after <= before {
			t.Fatalf("collection attempt %d committed no progress (%d holds): %v", attempt, after, err)
		}
	}
}

func countFixtureHolds(t *testing.T, f deletionImportFixture) int {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func assertFixtureHolds(t *testing.T, f deletionImportFixture, want int) {
	t.Helper()
	if got := countFixtureHolds(t, f); got != want {
		t.Fatalf("retained %d/%d proposals", got, want)
	}
}

// writeFixtureEvidence writes count exact evidence revisions into the
// fixture's source Run, in canonical reference order.
func writeFixtureEvidence(t *testing.T, f deletionImportFixture, count int) []ExactArtifact {
	t.Helper()
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	evidence := make([]ExactArtifact, count)
	for i := range evidence {
		data := []byte(fmt.Sprintf("evidence body %03d", i))
		written, err := runArtifacts.Write(f.ctx,
			artifacts.ArtifactRef{Namespace: "evidence", Name: fmt.Sprintf("item-%03d", i)},
			artifacts.Payload{MediaType: "text/plain", Data: data}, nil)
		if err != nil {
			t.Fatal(err)
		}
		evidence[i] = ExactArtifact{
			Ref: written.Ref, Digest: auditdomain.DigestBytes(data),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}
	}
	return evidence
}

// writeLostFixtureEvidence writes an evidence revision of the fixture's Run
// that receipts may cite before loseFixtureEvidence removes it. Its binding
// advances once, so the cited revision is not the current one.
func writeLostFixtureEvidence(t *testing.T, f deletionImportFixture) ExactArtifact {
	t.Helper()
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	target := artifacts.ArtifactRef{Namespace: "evidence", Name: "lost"}
	data := []byte("lost evidence body")
	written, err := runArtifacts.Write(f.ctx, target, artifacts.Payload{MediaType: "text/plain", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runArtifacts.Write(f.ctx, target,
		artifacts.Payload{MediaType: "text/plain", Data: []byte("replacement")}, written.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	return ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(data), MediaType: written.MediaType, SizeBytes: written.Size}
}

// loseFixtureEvidence removes a cited evidence revision. A receipt holds its
// evidence (DEL-R65), so only a purge of the source Run's scope removes it in
// production, and the fixture deletes the row under that purge's setting;
// the collection must still reject the receipt alone.
func loseFixtureEvidence(t *testing.T, f deletionImportFixture, evidence ExactArtifact) {
	t.Helper()
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(f.ctx, `SELECT set_config('contractor.lifecycle_purge', 'run', true)`); err != nil {
			return err
		}
		removed, err := tx.Exec(f.ctx, `DELETE FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = $2 AND name = $3 AND revision = $4`,
			f.request.RunID, evidence.Ref.Namespace, evidence.Ref.Name, *evidence.Ref.Revision)
		if err == nil && removed.RowsAffected() != 1 {
			err = errors.New("the evidence revision does not exist")
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
}

// A receipt holds its proposal and evidence strictly (DEL-R65): recording one
// whose evidence revision is gone from a stored Run scope, as when history
// retention removes it after the intake proved it, is the intake's not-found
// outcome ErrAccessDenied (FAIL-DEL-7), and records nothing.
func TestPostgresReceiptCitingARemovedRevisionIsAccessDenied(t *testing.T) {
	f := newDeletionImportFixture(t)
	lost := writeLostFixtureEvidence(t, f)
	loseFixtureEvidence(t, f, lost)
	if _, err := recordAuditChildReceipt(t, f, "removed-evidence", []ExactArtifact{lost}); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("receipt citing a removed revision = %v", err)
	}
	if _, err := readReceiptByID(f.ctx, f.pool, "removed-evidence-receipt"); err == nil {
		t.Fatal("the rejected receipt was recorded")
	}
}

// collectionTransactionImportBound is the most artifact imports one bounded
// retention transaction performs: the work bound plus the one proposal, with
// its maximum evidence, that may cross it.
const collectionTransactionImportBound = auditdomain.MaximumCollectionTransactionWork + 1 + MaxEvidenceRefs

// collectionTransactionTracer observes the retention transactions of a
// single-connection actor pool: how many committed and the most artifact
// imports one of them performed. With interrupt set, it cancels the attempt
// right after the next retention transaction commits. With endAttempt set,
// it cancels the attempt at its first import beyond
// collectionTransactionImportBound.
type collectionTransactionTracer struct {
	mu             sync.Mutex
	imports        int
	maxImports     int
	committed      int
	interrupt      context.CancelFunc
	endAttempt     context.CancelFunc
	attemptImports int
}

type collectionCommitKey struct{}

func (c *collectionTransactionTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	statement := strings.ToLower(strings.TrimSpace(data.SQL))
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case strings.HasPrefix(statement, "begin"), statement == "rollback":
		c.imports = 0
	// The fork statement takes its lineage kind as an argument.
	case strings.Contains(statement, "insert into artifact_lineage") &&
		slices.Contains(data.Args, any(artifacts.LineageAuditImport)):
		c.imports++
		c.attemptImports++
		if c.endAttempt != nil && c.attemptImports > collectionTransactionImportBound {
			// The import starts with a cancelled context, so it fails and
			// its transaction rolls back.
			c.endAttempt()
			c.endAttempt = nil
		}
	case statement == "commit":
		return context.WithValue(ctx, collectionCommitKey{}, c.imports)
	}
	return ctx
}

func (c *collectionTransactionTracer) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	imports, commit := ctx.Value(collectionCommitKey{}).(int)
	if !commit || imports == 0 || data.Err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.committed++
	c.maxImports = max(c.maxImports, imports)
	if c.interrupt != nil {
		c.interrupt()
		c.interrupt = nil
	}
}

// limitAttempt starts an attempt that endAttempt cancels once its imports
// exceed collectionTransactionImportBound; nil ends the limit.
func (c *collectionTransactionTracer) limitAttempt(endAttempt context.CancelFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.endAttempt, c.attemptImports = endAttempt, 0
}

// assertBounded checks that no retention transaction exceeded the work bound
// by more than the one proposal that may cross it.
func (c *collectionTransactionTracer) assertBounded(t *testing.T) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.committed == 0 || c.maxImports > collectionTransactionImportBound {
		t.Fatalf("%d retention transaction(s), largest imported %d artifacts", c.committed, c.maxImports)
	}
}

type multiQueryTracer []pgx.QueryTracer

func (m multiQueryTracer) TraceQueryStart(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	for _, tracer := range m {
		ctx = tracer.TraceQueryStart(ctx, conn, data)
	}
	return ctx
}

func (m multiQueryTracer) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	for _, tracer := range m {
		tracer.TraceQueryEnd(ctx, conn, data)
	}
}

func collectFixtureProposals(ctx context.Context, f deletionImportFixture, sourceSucceeded bool) error {
	query := ListQuery{Limit: 200}
	ctx = WithCollectionDirectVerificationCache(ctx)
	for {
		page, err := f.intake.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, query)
		if err != nil {
			return err
		}
		pending := make([]ImportRequest, 0, len(page))
		for _, candidate := range page {
			if !candidate.NeedsRetention(sourceSucceeded) {
				continue
			}
			request := f.request
			request.Proposal = candidate.Receipt.Proposal.Ref
			if candidate.ReadFailure != "" {
				if err := f.intake.RejectAuditCollection(ctx, request, candidate.ReadFailure); err != nil {
					return err
				}
				continue
			}
			pending = append(pending, request)
		}
		if len(pending) != 0 {
			if err := f.intake.RetainAuditCollectionBatch(ctx, pending); err != nil {
				return err
			}
		}
		if len(page) < query.Limit {
			return nil
		}
		last := page[len(page)-1].Receipt
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
	}
}

// A proposal whose own evidence cannot be retained is rejected alone: one
// with a missing evidence revision and one whose evidence no longer matches
// its receipt leave the other proposals of the page collectable.
func TestPostgresCollectionRejectsOnlyUnretainableProposals(t *testing.T) {
	f := newDeletionImportFixture(t)
	evidence := writeFixtureEvidence(t, f, 2)
	missing := writeLostFixtureEvidence(t, f)
	corrupt := evidence[1]
	corrupt.SizeBytes++
	unretainable := map[string]string{
		insertAuditChildReceiptWithEvidence(t, f, "missing-evidence", []ExactArtifact{evidence[0], missing}).ReceiptID: "finding-proposal-artifact-missing",
		insertAuditChildReceiptWithEvidence(t, f, "corrupt-evidence", []ExactArtifact{evidence[0], corrupt}).ReceiptID: "finding-proposal-artifact-invalid",
	}
	for i := range 3 {
		insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("valid-candidate-%d", i), evidence)
	}
	loseFixtureEvidence(t, f, missing)
	if err := collectFixtureProposals(f.ctx, f, false); err != nil {
		t.Fatalf("collection with unretainable proposals: %v", err)
	}
	assertFixtureHolds(t, f, 4)
	rows, err := f.pool.Query(f.ctx, `
SELECT entity_id, summary ->> 'reason' FROM audit_events
 WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'`, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	rejected := map[string]string{}
	for rows.Next() {
		var receiptID, reason string
		if err := rows.Scan(&receiptID, &reason); err != nil {
			t.Fatal(err)
		}
		rejected[receiptID] = reason
	}
	if err := rows.Err(); err != nil || len(rejected) != len(unretainable) {
		t.Fatalf("rejected proposals = %v, %v; want %v", rejected, err, unretainable)
	}
	for receiptID, reason := range unretainable {
		if rejected[receiptID] != reason {
			t.Fatalf("rejected proposals = %v, want %v", rejected, unretainable)
		}
	}
	page, err := f.intake.ListAuditCollection(f.ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(page) != 6 {
		t.Fatalf("collection page = %d receipts, %v", len(page), err)
	}
	for _, candidate := range page {
		if candidate.NeedsRetention(false) {
			t.Fatalf("receipt %s still needs retention: %+v", candidate.Receipt.ReceiptID, candidate)
		}
	}
	// A replay over the settled page changes nothing.
	audit, err := auditstoretest.NewStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if err := collectFixtureProposals(f.ctx, f, false); err != nil {
		t.Fatal(err)
	}
	replayed, err := auditstoretest.NewStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil || replayed.Revision != audit.Revision {
		t.Fatalf("collection replay changed the Audit: %+v, %v", replayed, err)
	}
}

// Rejecting a proposal replays the transaction that failed on it. Direct
// assessments that fit the evidence budget in that rolled-back transaction
// must still be accepted by the replay.
func TestPostgresCollectionRejectionKeepsEarlierDirectAssessments(t *testing.T) {
	f := newDirectCollectionFixture(t)
	missing := writeLostFixtureEvidence(t, f)
	keys := []string{"direct-0", "direct-1", "direct-2", "missing-evidence", "direct-3"}
	verifications := make([]auditdomain.DirectVerificationResult, len(keys))
	var rejectedID string
	for index, key := range keys {
		cited := []ExactArtifact{}
		if key == "missing-evidence" {
			cited = []ExactArtifact{missing}
		}
		receipt := insertAuditChildReceiptWithEvidence(t, f, key, cited)
		if key == "missing-evidence" {
			rejectedID = receipt.ReceiptID
		}
		verifications[index] = auditdomain.DirectVerificationResult{
			InvocationID: key + "-invocation", ClientKey: key,
			Assessment: "supported", Summary: "Verified exact candidate.", EvidenceIDs: []string{},
		}
	}
	loseFixtureEvidence(t, f, missing)
	sort.Slice(verifications, func(i, j int) bool { return verifications[i].ClientKey < verifications[j].ClientKey })
	encoded, err := json.Marshal(auditdomain.DirectVerificationSet{
		Schema: auditdomain.DirectVerificationsSchema, Verifications: verifications,
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	runArtifacts, err := artifactService.Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Write(f.ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "direct-result"},
		artifacts.Payload{MediaType: auditdomain.DirectVerificationsMediaType, Data: encoded}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(f.ctx, artifactService, f.request.RunID, "result", output.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(f.ctx, artifactService, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(f.pool).TransitionRun(f.ctx, f.request.RunID,
		runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "collection_fixture_succeeded"}); err != nil {
		t.Fatal(err)
	}
	// Admit exactly two direct assessments: the third verified proposal finds
	// the budget exhausted before the transaction fails on the missing evidence.
	base, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := prepareDirectVerification(f.ctx, tx, artifacts.NewService(artifacts.NewPostgresRepository(tx)),
		directVerificationInput{
			AuditID: f.request.AuditID, ProjectID: "delete-project", RunID: f.request.RunID,
			WorkflowClosureDigest: base.Origin.Workflow.ClosureDigest,
		})
	if rollbackErr := tx.Rollback(f.ctx); err != nil || rollbackErr != nil || prepared == nil {
		t.Fatalf("prepare direct verification = (%+v, %v, %v)", prepared, err, rollbackErr)
	}
	charge := prepared.descriptor.Size + int64(len(prepared.contractBytes))
	if _, err := f.pool.Exec(f.ctx, `UPDATE audits SET max_evidence_bytes = $2 WHERE audit_id = $1`,
		f.request.AuditID, 2*charge+charge/2); err != nil {
		t.Fatal(err)
	}

	if err := collectFixtureProposals(f.ctx, f, true); err != nil {
		t.Fatalf("collection with one unretainable proposal: %v", err)
	}
	assertFixtureHolds(t, f, len(keys))
	inbox, err := f.intake.ListAuditInbox(f.ctx, f.request.OwnerID, f.request.AuditID, ListQuery{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	var assessed []string
	for _, receipt := range inbox {
		if receipt.ProjectProposal == nil {
			continue
		}
		if len(readDirectHistoryForTest(t, f.ctx, f.pool, f.request.OwnerID, "delete-project", receipt.ReceiptID, f.request.AuditID)) != 0 {
			assessed = append(assessed, receipt.ClientKey)
		}
	}
	sort.Strings(assessed)
	if strings.Join(assessed, ",") != "direct-0,direct-1" {
		t.Fatalf("retained direct results=%v", assessed)
	}
	var rejected bool
	if err := f.pool.QueryRow(f.ctx, `
SELECT EXISTS (SELECT 1 FROM audit_events
 WHERE audit_id = $1 AND kind = 'finding.proposal_rejected' AND entity_id = $2)`,
		f.request.AuditID, rejectedID).Scan(&rejected); err != nil || !rejected {
		t.Fatalf("missing-evidence proposal rejection = (%t, %v)", rejected, err)
	}
}
