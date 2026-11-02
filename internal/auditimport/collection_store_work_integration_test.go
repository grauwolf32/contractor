//go:build integration

package auditimport

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The accepted collection bounds: 64 items (auditstore.MaxCollectionItems)
// whose result and evidence links fill one receipt's 1,024 links
// (auditstore.MaxArtifactLinksPerCall). Two evidence records repeat an earlier
// revision, so 958 distinct evidence revisions and the result are retained.
const (
	maximumCollectionItems    = auditstore.MaxCollectionItems
	maximumEvidencePerItem    = (auditstore.MaxArtifactLinksPerCall - maximumCollectionItems) / maximumCollectionItems
	maximumRepeatedEvidence   = 2
	maximumDistinctEvidence   = maximumCollectionItems*maximumEvidencePerItem - maximumRepeatedEvidence
	maximumRetainedRevisions  = maximumDistinctEvidence + 1
	maximumCollectionLinks    = maximumCollectionItems * (1 + maximumEvidencePerItem)
	maximumDescriptionBatches = (maximumDistinctEvidence + artifacts.MaxExactReadBatchSize - 1) / artifacts.MaxExactReadBatchSize
	maximumRetentionBatches   = (maximumRetainedRevisions + auditdomain.MaximumCollectionTransactionWork - 1) /
		auditdomain.MaximumCollectionTransactionWork
)

// An accepted collection at the maximum accepted size reads its evidence
// descriptors without payloads in bounded batches and retains every distinct
// revision in bounded artifact-write transactions, each holding the claim and
// the Audit row, before the one receipt links them (AUX-R98).
func TestPostgresMaximumCollectionDescribesEvidenceAndRetainsInBoundedTransactions(t *testing.T) {
	pool := completionPool(t)
	f := newCompletionFixtureWithOptions(t, pool, completionOptions{
		acceptance: "automatic", items: maximumCollectionItems,
	})
	revisions := f.writeEvidenceResult(t)
	execution, snapshot := f.terminal(t, runstore.RunSucceeded)
	counter := &collectionStatementCounter{}
	importer := f.tracedImporter(t, counter)

	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("maximum collection = (%t, %v)", changed, err)
	}
	counts := counter.snapshot()
	want := collectionStatementCounts{
		evidencePayloadReads: 0, evidenceDescriptions: maximumDescriptionBatches,
		artifactWrites: maximumRetentionBatches, forks: maximumRetainedRevisions,
		artifactWriteStatements: artifactWriteStatements(maximumRetentionBatches, maximumRetainedRevisions, 0),
	}
	if counts != want {
		t.Fatalf("maximum collection statements = %+v, want %+v", counts, want)
	}
	f.requireMaximumCollectionLinks(t, execution.ExecutionID, revisions)
}

// A collection interrupted after its first artifact-write transaction
// committed, as an expiring deadline or a lost acknowledgement leaves it,
// links nothing and keeps the execution collecting. The next attempt replays
// the committed retentions, retains only the rest and links exactly what an
// uninterrupted collection links.
func TestPostgresMaximumCollectionResumesAfterInterruptedRetention(t *testing.T) {
	pool := completionPool(t)
	f := newCompletionFixtureWithOptions(t, pool, completionOptions{
		acceptance: "automatic", items: maximumCollectionItems,
	})
	revisions := f.writeEvidenceResult(t)
	execution, snapshot := f.terminal(t, runstore.RunSucceeded)

	attempt, cancel := context.WithCancel(f.ctx)
	defer cancel()
	interrupted := &collectionStatementCounter{interrupt: cancel}
	changed, err := f.tracedImporter(t, interrupted).Collect(attempt, f.claim, snapshot, execution)
	if changed || !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted collection = (%t, %v), want context.Canceled", changed, err)
	}
	if counts := interrupted.snapshot(); counts.artifactWrites != 1 || counts.forks != auditdomain.MaximumCollectionTransactionWork ||
		counts.artifactWriteStatements != artifactWriteStatements(1, auditdomain.MaximumCollectionTransactionWork, 0) {
		t.Fatalf("interrupted collection statements = %+v, want one full artifact-write transaction", counts)
	}
	f.requireCollecting(t, execution.ExecutionID)
	committed := f.retainedBindings(t)
	if len(committed) != auditdomain.MaximumCollectionTransactionWork {
		t.Fatalf("interrupted collection kept %d retained bindings, want %d",
			len(committed), auditdomain.MaximumCollectionTransactionWork)
	}
	var links int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id = $1`, f.id).Scan(&links))
	if links != 0 {
		t.Fatalf("interrupted collection linked %d artifacts", links)
	}

	resumed := &collectionStatementCounter{}
	changed, err = f.tracedImporter(t, resumed).Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("resumed collection = (%t, %v)", changed, err)
	}
	want := collectionStatementCounts{
		evidenceDescriptions: maximumDescriptionBatches,
		artifactWrites:       maximumRetentionBatches, forks: maximumRetainedRevisions,
		artifactWriteStatements: artifactWriteStatements(maximumRetentionBatches,
			maximumRetainedRevisions-auditdomain.MaximumCollectionTransactionWork,
			auditdomain.MaximumCollectionTransactionWork),
	}
	if counts := resumed.snapshot(); counts != want {
		t.Fatalf("resumed collection statements = %+v, want %+v", counts, want)
	}
	retained := f.retainedBindings(t)
	if len(retained) != maximumRetainedRevisions {
		t.Fatalf("resumed collection retained %d bindings, want %d", len(retained), maximumRetainedRevisions)
	}
	for name, revision := range committed {
		if retained[name] != revision {
			t.Fatalf("replay replaced retained binding %s revision %s with %s", name, revision, retained[name])
		}
	}
	f.requireMaximumCollectionLinks(t, execution.ExecutionID, revisions)
}

// collectionStatementCounts classifies the artifact statements of one
// collection attempt. artifactWriteStatements counts every statement of the
// committed artifact-write transactions, from begin to commit.
// A transient evidence description failure leaves the execution collecting
// before anything is retained. A claim lost between two artifact-write
// transactions stops retention at ErrClaimLost with the earlier transaction
// committed but unlinked, and the next holder replays it and links exactly
// what an uninterrupted collection links.
func TestPostgresMaximumCollectionStopsAtTransientFailureAndClaimLoss(t *testing.T) {
	pool := completionPool(t)
	f := newCompletionFixtureWithOptions(t, pool, completionOptions{
		acceptance: "automatic", items: maximumCollectionItems,
	})
	revisions := f.writeEvidenceResult(t)
	execution, snapshot := f.terminal(t, runstore.RunSucceeded)

	unavailable := errors.New("artifact metadata unavailable")
	faulty := &collectionStatementCounter{}
	changed, err := f.tracedImporter(t, faulty, func(access ArtifactAccess) ArtifactAccess {
		return describeFaultAccess{ArtifactAccess: access, err: unavailable}
	}).Collect(f.ctx, f.claim, snapshot, execution)
	if changed || !errors.Is(err, unavailable) || errors.Is(err, ErrPermanent) {
		t.Fatalf("collection with unavailable evidence metadata = (%t, %v)", changed, err)
	}
	if counts := faulty.snapshot(); counts.artifactWrites != 0 || counts.forks != 0 {
		t.Fatalf("collection with unavailable evidence metadata wrote artifacts: %+v", counts)
	}
	f.requireCollecting(t, execution.ExecutionID)
	if retained := f.retainedBindings(t); len(retained) != 0 {
		t.Fatalf("collection with unavailable evidence metadata retained %d bindings", len(retained))
	}

	lost := &collectionStatementCounter{}
	lost.interrupt = func() { mustCompletion(t, f.audits.ReleaseClaim(context.WithoutCancel(f.ctx), f.claim)) }
	changed, err = f.tracedImporter(t, lost).Collect(f.ctx, f.claim, snapshot, execution)
	if changed || !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("collection after claim loss = (%t, %v), want ErrClaimLost", changed, err)
	}
	// The second transaction locks the claim first and finds it lost.
	if counts := lost.snapshot(); counts.artifactWrites != 1 || counts.forks != auditdomain.MaximumCollectionTransactionWork {
		t.Fatalf("collection after claim loss statements = %+v", counts)
	}
	f.requireCollecting(t, execution.ExecutionID)
	committed := f.retainedBindings(t)
	if len(committed) != auditdomain.MaximumCollectionTransactionWork {
		t.Fatalf("collection after claim loss kept %d retained bindings, want %d",
			len(committed), auditdomain.MaximumCollectionTransactionWork)
	}

	claims, err := f.audits.Claim(f.ctx, auditstore.ClaimParams{HolderID: f.id + "-next", Lease: 5 * time.Minute, Limit: 100})
	mustCompletion(t, err)
	var next auditstore.ControllerClaim
	for _, claim := range claims {
		if claim.AuditID == f.id {
			next = claim
		}
	}
	if next.AuditID == "" || next.Epoch <= f.claim.Epoch {
		t.Fatalf("next holder claims = %+v", claims)
	}
	snapshot, err = f.audits.GetReconcileSnapshot(f.ctx, next)
	mustCompletion(t, err)
	resumed := &collectionStatementCounter{}
	changed, err = f.tracedImporter(t, resumed).Collect(f.ctx, next, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("collection by the next holder = (%t, %v)", changed, err)
	}
	if counts := resumed.snapshot(); counts.artifactWrites != maximumRetentionBatches ||
		counts.artifactWriteStatements != artifactWriteStatements(maximumRetentionBatches,
			maximumRetainedRevisions-auditdomain.MaximumCollectionTransactionWork,
			auditdomain.MaximumCollectionTransactionWork) {
		t.Fatalf("collection by the next holder statements = %+v", counts)
	}
	retained := f.retainedBindings(t)
	for name, revision := range committed {
		if retained[name] != revision {
			t.Fatalf("next holder replaced retained binding %s revision %s with %s", name, revision, retained[name])
		}
	}
	f.requireMaximumCollectionLinks(t, execution.ExecutionID, revisions)
}

type collectionStatementCounts struct {
	evidencePayloadReads, evidenceDescriptions, artifactWrites, forks, artifactWriteStatements int
}

// artifactWriteStatements is the documented statement count of retaining
// fresh and replayed revisions in a number of artifact-write transactions:
// begin, claim lock, Audit lock, claim check and commit per transaction, two
// statements per new retention and five per replayed one, which verifies the
// retained binding's content and lineage (AUX-R98).
func artifactWriteStatements(transactions, fresh, replayed int) int {
	return 5*transactions + 2*fresh + 5*replayed
}

// collectionStatementCounter classifies the artifact statements of one
// collection attempt. interrupt, when set, cancels the attempt once its first
// artifact-write transaction has committed.
type collectionStatementCounter struct {
	mu        sync.Mutex
	counts    collectionStatementCounts
	open      map[*pgx.Conn]int
	locked    map[*pgx.Conn]bool
	interrupt context.CancelFunc
}

type artifactWriteEnd struct{}

func (c *collectionStatementCounter) TraceQueryStart(
	ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.open == nil {
		c.open, c.locked = map[*pgx.Conn]int{}, map[*pgx.Conn]bool{}
	}
	statement := strings.ToLower(strings.TrimSpace(data.SQL))
	if strings.HasPrefix(statement, "begin") {
		c.open[conn], c.locked[conn] = 0, false
	}
	if _, open := c.open[conn]; open {
		c.open[conn]++
	}
	switch {
	case strings.Contains(data.SQL, "blob.payload") && len(data.Args) >= 3 && data.Args[2] == "evidence":
		c.counts.evidencePayloadReads++
	case strings.Contains(data.SQL, "AS input(namespace, name, revision, ordinality)"):
		c.counts.evidenceDescriptions++
	case strings.Contains(data.SQL, "FROM audits") && strings.Contains(data.SQL, "state <> 'deleting'") &&
		strings.Contains(data.SQL, "FOR UPDATE"):
		c.counts.artifactWrites++
		c.locked[conn] = true
	case strings.Contains(data.SQL, "WITH source_selection AS MATERIALIZED"):
		c.counts.forks++
	case statement == "commit" || statement == "rollback":
		if c.locked[conn] {
			return context.WithValue(ctx, artifactWriteEnd{}, statement == "commit")
		}
		delete(c.open, conn)
	}
	return ctx
}

func (c *collectionStatementCounter) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	committed, end := ctx.Value(artifactWriteEnd{}).(bool)
	if !end {
		return
	}
	c.mu.Lock()
	var interrupt context.CancelFunc
	if committed && data.Err == nil {
		c.counts.artifactWriteStatements += c.open[conn]
		interrupt, c.interrupt = c.interrupt, nil
	}
	delete(c.open, conn)
	delete(c.locked, conn)
	c.mu.Unlock()
	if interrupt != nil {
		interrupt()
	}
}

func (c *collectionStatementCounter) snapshot() collectionStatementCounts {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.counts
}

// tracedImporter returns the production importer whose artifact reads and
// claimed writes use a pool of the fixture database traced by counter,
// optionally wrapped to inject faults.
func (f *completionFixture) tracedImporter(
	t *testing.T, counter *collectionStatementCounter, wrap ...func(ArtifactAccess) ArtifactAccess,
) *Importer {
	t.Helper()
	config := f.pool.Config()
	config.ConnConfig.Tracer = counter
	traced, err := pgxpool.NewWithConfig(f.ctx, config)
	mustCompletion(t, err)
	t.Cleanup(traced.Close)
	var access ArtifactAccess
	access, err = NewArtifactAccess(artifacts.NewService(artifacts.NewPostgresRepository(traced)), hookstore.NewClaimedArtifacts(traced))
	mustCompletion(t, err)
	for _, apply := range wrap {
		access = apply(access)
	}
	importer, err := New(f.audits, f.audits, f.runs, access)
	mustCompletion(t, err)
	return importer
}

// describeFaultAccess fails every evidence description with err.
type describeFaultAccess struct {
	ArtifactAccess
	err error
}

func (a describeFaultAccess) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	a.ArtifactAccess = a.ArtifactAccess.WithClaim(claim)
	return a
}

func (a describeFaultAccess) DescribeRunExact(
	context.Context, string, []contracts.ArtifactRef,
) ([]auditstore.ExactArtifact, error) {
	return nil, a.err
}

// writeEvidenceResult publishes the Worker's result package in which every
// manifest item cites maximumEvidencePerItem external evidence records of
// the source Run, and returns the distinct evidence revisions they cite.
// Record i cites revision i modulo maximumDistinctEvidence.
func (f *completionFixture) writeEvidenceResult(t *testing.T) []auditstore.ExactArtifact {
	t.Helper()
	project, err := f.artifacts.Project(f.id)
	mustCompletion(t, err)
	stored, err := project.Read(f.ctx, f.manifest.Ref)
	mustCompletion(t, err)
	manifest, err := auditdomain.DecodeExecutionManifest(stored.Payload.Data)
	mustCompletion(t, err)
	if len(manifest.Items) != maximumCollectionItems {
		t.Fatalf("manifest has %d items, want %d", len(manifest.Items), maximumCollectionItems)
	}
	store, err := f.artifacts.Run(f.run.RunID)
	mustCompletion(t, err)
	revisions := make([]auditstore.ExactArtifact, maximumDistinctEvidence)
	for index := range revisions {
		data := []byte(fmt.Sprintf("source trace %04d of %s", index, f.id))
		written, err := store.Write(f.ctx, contracts.ArtifactRef{Namespace: "evidence", Name: fmt.Sprintf("trace-%04d", index)},
			artifacts.Payload{MediaType: "text/plain", Data: data}, nil)
		mustCompletion(t, err)
		revisions[index] = contracts.ExactContent(written.Ref, "text/plain", data)
	}
	records := make([]auditdomain.Evidence, 0, maximumCollectionItems*maximumEvidencePerItem)
	results := make([]auditdomain.CheckResult, 0, len(manifest.Items))
	for itemIndex, item := range manifest.Items {
		ids := make([]string, maximumEvidencePerItem)
		for offset := range ids {
			index := itemIndex*maximumEvidencePerItem + offset
			ids[offset] = fmt.Sprintf("ev-%04d", index)
			ref := revisions[index%maximumDistinctEvidence].Ref
			records = append(records, auditdomain.Evidence{
				ID: ids[offset], Kind: "source", Summary: "Source trace of the check.", Artifact: &ref,
			})
		}
		results = append(results, auditdomain.CheckResult{
			ItemKey: item.ItemKey, SubjectKey: item.SubjectKey, Assessment: "inconclusive",
			Summary: "The source trace was not completed.", EvidenceIDs: ids,
			Coverage: auditdomain.ResultCoverage{
				Requested: []string{"source-trace"}, Completed: []string{}, Gaps: []string{},
			},
			Proposals: []auditdomain.ProposalSelection{},
		})
	}
	resultSet, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: f.manifest.Digest, Results: results,
	})
	mustCompletion(t, err)
	evidence, err := auditdomain.EncodeEvidence(auditdomain.EvidenceEnvelope{
		Schema: auditdomain.EvidenceSchema, Evidence: records,
	})
	mustCompletion(t, err)
	payload, _, err := auditdomain.BuildPackage("result-package", auditdomain.PackageKindCheckResults, "",
		[]auditdomain.PackageInput{
			{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: resultSet},
			{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: auditdomain.JSONMediaType, Data: evidence},
		})
	mustCompletion(t, err)
	target := f.run.AuditCompletion.Contract.ResultArtifact
	_, err = store.Write(f.ctx, contracts.ArtifactRef{Namespace: target.Namespace, Name: target.Name},
		artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload}, nil)
	mustCompletion(t, err)
	return revisions
}

// retainedBindings returns the revision of every binding retained in the
// Audit's namespace of the fixture Project, by name.
func (f *completionFixture) retainedBindings(t *testing.T) map[string]string {
	t.Helper()
	rows, err := f.pool.Query(f.ctx, `
SELECT name, current_revision FROM artifact_bindings
 WHERE scope_kind = 'project' AND scope_id = $1 AND namespace = $2`, f.id, auditdomain.ArtifactNamespace(f.id))
	mustCompletion(t, err)
	defer rows.Close()
	result := map[string]string{}
	for rows.Next() {
		var name, revision string
		mustCompletion(t, rows.Scan(&name, &revision))
		result[name] = revision
	}
	mustCompletion(t, rows.Err())
	return result
}

// requireMaximumCollectionLinks fails unless the accepted receipt links every
// result and evidence record to its retained copy: records citing one
// revision share the copy named by the first sorted citing record, each copy
// holds its source's content, and the Audit is charged each distinct
// revision once.
func (f *completionFixture) requireMaximumCollectionLinks(
	t *testing.T, executionID string, revisions []auditstore.ExactArtifact,
) {
	t.Helper()
	var disposition string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT collection_disposition FROM audit_executions WHERE execution_id = $1`, executionID).Scan(&disposition))
	if disposition != string(auditstore.CollectionAccepted) {
		t.Fatalf("collection disposition = %s, want accepted", disposition)
	}
	rows, err := f.pool.Query(f.ctx, `
SELECT logical_key, artifact_ref->>'name', artifact_ref->>'revision', artifact_digest, size_bytes
  FROM audit_artifact_links WHERE audit_id = $1`, f.id)
	mustCompletion(t, err)
	defer rows.Close()
	retained := f.retainedBindings(t)
	copies := map[string]struct{}{}
	var resultSize int64
	links := 0
	for rows.Next() {
		var key, name, revision, digest string
		var size int64
		mustCompletion(t, rows.Scan(&key, &name, &revision, &digest, &size))
		links++
		copies[name+"@"+revision] = struct{}{}
		if retained[name] != revision {
			t.Fatalf("link %s names %s@%s, not the retained binding", key, name, revision)
		}
		if strings.HasPrefix(key, "result/") {
			if name != auditdomain.DeterministicID("result", executionID) {
				t.Fatalf("result link %s names %s", key, name)
			}
			resultSize = size
			continue
		}
		var index int
		if _, err := fmt.Sscanf(key, "evidence/"+executionID+"/ev-%04d", &index); err != nil {
			t.Fatalf("unexpected link %s", key)
		}
		source := revisions[index%maximumDistinctEvidence]
		owner := fmt.Sprintf("ev-%04d", index%maximumDistinctEvidence)
		if name != auditdomain.DeterministicID("evidence", executionID, owner) ||
			digest != source.Digest || size != source.SizeBytes {
			t.Fatalf("evidence link %s = %s %s %d, want the copy of %+v named by %s", key, name, digest, size, source, owner)
		}
	}
	mustCompletion(t, rows.Err())
	if links != maximumCollectionLinks || len(copies) != maximumRetainedRevisions {
		t.Fatalf("receipt links %d artifacts through %d copies, want %d through %d",
			links, len(copies), maximumCollectionLinks, maximumRetainedRevisions)
	}
	charged := resultSize
	for _, revision := range revisions {
		charged += revision.SizeBytes
	}
	var retainedBytes int64
	mustCompletion(t, f.pool.QueryRow(f.ctx, `SELECT retained_evidence_bytes FROM audits WHERE audit_id = $1`, f.id).Scan(&retainedBytes))
	if retainedBytes != charged {
		t.Fatalf("Audit charged %d retained bytes, want %d", retainedBytes, charged)
	}
}
