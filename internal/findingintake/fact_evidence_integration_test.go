//go:build integration

package findingintake

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func TestPostgresFactEvidenceEqualityUsesExistingCorrelationAndExactSelection(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	const owner = "fact-key-owner"
	projects := projectstore.NewPostgresStore(pool)
	for _, project := range []string{"fact-key-a", "fact-key-b"} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{OwnerID: owner, ProjectID: project, Kind: projectstore.KindProject, Name: project, IdempotencyKey: project, RequestDigest: auditdomain.DigestBytes([]byte(project))}); err != nil {
			t.Fatal(err)
		}
	}
	grants := []controlplane.AllocationGrant{quotaSubmissionGrant(t, ctx, pool, owner, "fact-key-a", "fact-key-run-0"), quotaSubmissionGrant(t, ctx, pool, owner, "fact-key-a", "fact-key-run-1"), quotaSubmissionGrant(t, ctx, pool, owner, "fact-key-b", "fact-key-run-2")}
	store := artifacts.NewService(storage.Repository(pool))
	service, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	write := func(index int, name, text, media string) contracts.ArtifactRef {
		t.Helper()
		scope, _ := store.Run(grants[index].RunID)
		saved, err := scope.Write(ctx, contracts.ArtifactRef{Namespace: grants[index].Namespace, Name: name}, artifacts.Payload{MediaType: media, Data: []byte(text)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return saved.Ref
	}
	a := write(0, "a", "nginx observed", "text/plain")
	b := write(0, "b", "condition observed", "text/plain")
	equalA := write(1, "independent-a", "nginx observed", "application/octet-stream")
	equalB := write(1, "independent-b", "condition observed", "text/plain")
	foreignA := write(2, "a", "nginx observed", "text/plain")
	foreignB := write(2, "b", "condition observed", "text/plain")
	inputs := []Submission{testSubmission("fact-key-invocation", "first", []contracts.ArtifactRef{a, b}), testSubmission("fact-key-invocation", "second", []contracts.ArtifactRef{equalB, equalA})}
	for i := range inputs {
		inputs[i].Proposal.SeveritySuggestion = "informational"
		inputs[i].Proposal.Title = fmt.Sprintf("Different claim %d", i)
	}
	receipts := make([]Receipt, 2)
	failures := make([]error, 2)
	var concurrent sync.WaitGroup
	start := make(chan struct{})
	for i := range inputs {
		concurrent.Go(func() { <-start; receipts[i], _, failures[i] = service.Submit(ctx, grants[i], inputs[i]) })
	}
	close(start)
	concurrent.Wait()
	for i := range receipts {
		if failures[i] != nil || receipts[i].FindingID == nil {
			t.Fatalf("receipt %d: %+v %v", i, receipts[i], failures[i])
		}
	}
	finding := *receipts[0].FindingID
	if finding != *receipts[1].FindingID || receipts[0].ReceiptID == receipts[1].ReceiptID {
		t.Fatal("equal complete sets did not share one Finding and retain separate receipts")
	}
	for i := range inputs {
		replayed, ok, err := service.Submit(ctx, grants[i], inputs[i])
		if err != nil || !ok || replayed.ReceiptID != receipts[i].ReceiptID {
			t.Fatal(replayed, ok, err)
		}
	}
	submit := func(index int, key, severity string, refs ...contracts.ArtifactRef) Receipt {
		t.Helper()
		input := testSubmission("fact-key-later", key, refs)
		input.Proposal.Title = key
		input.Proposal.SeveritySuggestion = severity
		receipt, _, err := service.Submit(ctx, grants[index], input)
		if err != nil || receipt.FindingID == nil {
			t.Fatalf("%s: %+v %v", key, receipt, err)
		}
		return receipt
	}
	from, _ := artifacts.ProjectScope("fact-key-a")
	to, _ := artifacts.RunScope(grants[1].RunID)
	fork, err := store.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageInputFork, Source: from, SourceRef: receipts[0].ProjectEvidence[0].Ref, SourceFrozen: true, Target: to, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "forked-a"}, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	forked := submit(1, "forked", "informational", fork.TargetRef, equalB)
	if *forked.FindingID != finding {
		t.Fatal("exact fork did not reuse evidence key")
	}
	duplicateContent := write(0, "independent-same-a", "nginx observed", "text/plain")
	single := submit(0, "duplicate-content", "informational", a, duplicateContent)
	sameSingle := submit(1, "one-content", "informational", equalA)
	if *single.FindingID != *sameSingle.FindingID || *single.FindingID == finding {
		t.Fatal("digest set was treated as a multiset or overlap")
	}
	changed := write(1, "changed", "changed bytes", "text/plain")
	overlap := submit(1, "overlapping-set", "informational", equalA, changed)
	changedOnly := submit(1, "changed-only", "informational", changed)
	foreign := submit(2, "another-project", "informational", foreignA, foreignB)
	if *overlap.FindingID == finding || *changedOnly.FindingID == finding || *overlap.FindingID == *changedOnly.FindingID || *foreign.FindingID == finding {
		t.Fatal("changed, overlapping or foreign evidence merged")
	}
	firstIssue := submit(0, "ordinary-issue-one", "high", a, b)
	secondIssue := submit(0, "ordinary-issue-two", "high", a, b)
	if *firstIssue.FindingID == finding || *firstIssue.FindingID == *secondIssue.FindingID {
		t.Fatal("non-informational intake used coarse fact equality")
	}
	// The real receipt constraint rejects malformed fact variants before insertion.
	var columns []string
	if err := pool.QueryRow(ctx, `SELECT array_agg(attname::text ORDER BY attnum) FROM pg_attribute WHERE attrelid='finding_proposal_receipts'::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=''`).Scan(&columns); err != nil {
		t.Fatal(err)
	}
	projection := make([]string, len(columns))
	for i, column := range columns {
		columns[i] = pgx.Identifier{column}.Sanitize()
		projection[i] = "cloned." + columns[i]
	}
	cloneSQL := "INSERT INTO finding_proposal_receipts(" + strings.Join(columns, ",") + ") SELECT " + strings.Join(projection, ",") + " FROM finding_proposal_receipts AS seed CROSS JOIN LATERAL jsonb_populate_record(NULL::finding_proposal_receipts,to_jsonb(seed)||$2::jsonb) AS cloned WHERE seed.receipt_id=$1"
	for i, change := range []map[string]any{{"reason": "not_provided"}, {"adapter": "normalized-candidate/v1"}, {"value": nil}, {"availability": "unavailable"}, {}} {
		raw, _ := auditdomain.EncodeFindingFingerprint(receipts[0].Fingerprint)
		var document map[string]any
		_ = json.Unmarshal(raw, &document)
		for key, value := range change {
			document[key] = value
		}
		encoded, _ := json.Marshal(document)
		altered := map[string]any{"receipt_id": fmt.Sprintf("malformed-fact-%d", i), "submission_id": fmt.Sprintf("malformed-%d", i), "fingerprint_projection": string(encoded)}
		if i == 4 {
			report, _ := json.Marshal(inputs[0].Proposal)
			var proposal map[string]any
			_ = json.Unmarshal(report, &proposal)
			proposal["severity_suggestion"] = "high"
			report, _ = json.Marshal(proposal)
			altered["report_projection"] = string(report)
		}
		mutation, _ := json.Marshal(altered)
		if _, err := pool.Exec(ctx, cloneSQL, receipts[0].ReceiptID, string(mutation)); postgres.ConstraintName(err) != "finding_fact_fingerprint_shape" {
			t.Fatalf("malformed fact variant %d was not rejected by its constraint: %v", i, err)
		}
	}
	// Analyst edits change presentation, never the immutable producer key.
	severity := auditstore.SeverityHigh
	if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := ownerstore.NewTx(tx, storage).AssessProjectFinding(ctx, auditstore.ProjectFindingEdit{OwnerID: owner, ProjectID: "fact-key-a", FindingID: finding, ExpectedRevision: 1, Verdict: auditstore.VerdictTruePositive, Severity: &severity, Rationale: "Explicit assessment"})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	assessed := submit(1, "after-assessment", "informational", equalA, equalB)
	if *assessed.FindingID != finding {
		t.Fatal("assessment rekeyed fact")
	}
	current, err := querystore.New(pool, storage).GetProjectFinding(ctx, owner, "fact-key-a", finding)
	if err != nil || current.AnalystSeverity == nil || *current.AnalystSeverity != severity || current.Title == "after-assessment" {
		t.Fatal("later source changed first claim or analyst assessment", err)
	}
	// Explicit receipt selection captures only the chosen source of a merged Finding.
	consumer := grants[1]
	consumer.ReadPolicy = controlplane.ReadCurrentRun
	consumer.WritePolicy = controlplane.WriteInputsAndIntermediates
	consumer.FindingTools = []string{"select_project_findings"}
	publisher, err := NewCollectionPublisher(pool, projectCollectionReads{}, storage)
	if err != nil {
		t.Fatal(err)
	}
	selection := FindingSelection{ClientKey: "chosen-fact-source", Findings: []CollectionFindingSelection{}, ReceiptIDs: []string{receipts[0].ReceiptID}}
	selected, err := publisher.SelectAllocationFindings(ctx, consumer, selection)
	if err != nil || selected.EntryCount != 1 {
		t.Fatal(selected, err)
	}
	consumerScope, _ := store.Run(consumer.RunID)
	saved, err := consumerScope.Read(ctx, selected.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	collection, pkg, err := auditdomain.DecodeFindingCollectionPackage(saved.Payload.Data)
	if err != nil || len(collection.Entries) != 1 || collection.Entries[0].ReceiptID != receipts[0].ReceiptID {
		t.Fatal("selection expanded to other sources", err)
	}
	member, ok := pkg.MemberByID(collection.Entries[0].Evidence[0].DocumentID)
	if !ok || string(member.Data()) != "nginx observed" {
		t.Fatal("wrong exact selected proof")
	}
	// A newer source revision cannot substitute the captured source or its key.
	scope0, _ := store.Run(grants[0].RunID)
	if _, err := scope0.Write(ctx, contracts.ArtifactRef{Namespace: a.Namespace, Name: a.Name}, artifacts.Payload{MediaType: "text/plain", Data: []byte("new latest bytes")}, a.Revision); err != nil {
		t.Fatal(err)
	}
	selectedAgain, err := publisher.SelectAllocationFindings(ctx, consumer, selection)
	if err != nil || !selectedAgain.Replayed || !selectedAgain.Artifact.Ref.SameExact(selected.Artifact.Ref) {
		t.Fatal("selected source changed on replay", err)
	}
	// Invalid new evidence writes no receipt or proposal; Server supplies all hashes.
	var before int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts").Scan(&before); err != nil {
		t.Fatal(err)
	}
	for i, refs := range [][]contracts.ArtifactRef{nil, {foreignA}} {
		input := testSubmission("invalid-fact", fmt.Sprint(i), refs)
		input.Proposal.SeveritySuggestion = "informational"
		if _, _, err := service.Submit(ctx, grants[0], input); err == nil {
			t.Fatal("invalid evidence accepted")
		}
	}
	var after int
	_ = pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts").Scan(&after)
	if before != after {
		t.Fatal("invalid evidence retained a partial receipt")
	}
	runs := runstore.NewPostgresStore(pool, storage)
	for _, grant := range grants[:2] {
		if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
			t.Fatal(err)
		}
		if _, err := runs.TransitionRun(ctx, grant.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
			t.Fatal(err)
		}
		if err := deleteRun(ctx, pool, owner, grant.RunID); err != nil {
			t.Fatal(err)
		}
	}
	replayed, ok, err := service.Submit(ctx, grants[0], inputs[0])
	if err != nil || !ok || replayed.ReceiptID != receipts[0].ReceiptID {
		t.Fatal("replay reread deleted source", err)
	}
	retained, err := service.GetProjectReceipt(ctx, owner, "fact-key-a", receipts[0].ReceiptID)
	if err != nil || retained.Retention != RetentionProjectHeld || !retained.Origin.RunDeleted {
		t.Fatal(retained, err)
	}
	projectScope, _ := store.Project("fact-key-a")
	if proof, err := projectScope.Read(ctx, retained.ProjectEvidence[0].Ref); err != nil || string(proof.Payload.Data) != "nginx observed" {
		t.Fatal("retained evidence was replaced", err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: owner, ProjectID: "fact-key-a", ExpectedRevision: 1}); err != nil {
		t.Fatal(err)
	}
	done := false
	for range 100 {
		if err := postgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			gone, err := PurgeProjectReceipts(ctx, tx, "fact-key-a", storage)
			if err != nil || !gone {
				return err
			}
			purger, err := storage.Purger(tx)
			if err != nil {
				return err
			}
			done, err = purger.PurgeProject(ctx, "fact-key-a")
			return err
		}); err != nil {
			t.Fatal(err)
		}
		if done {
			break
		}
	}
	if !done {
		t.Fatal("merged fact blocked ordinary Project purge")
	}
	if _, err := service.GetProjectReceipt(ctx, owner, "fact-key-a", receipts[0].ReceiptID); !errors.Is(err, ErrNotFound) {
		t.Fatal("receipt survived purge", err)
	}
	var remains int
	if err := pool.QueryRow(ctx, "SELECT (SELECT count(*) FROM project_findings WHERE project_id='fact-key-a')+(SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id='fact-key-a')").Scan(&remains); err != nil || remains != 0 {
		t.Fatal("Project-owned data survived purge", remains, err)
	}
}
