package auditservice

import (
	"context"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Observe calls while the ordinary PostgreSQL repositories perform every read.
type observedCollectionEvidence struct {
	auditimport.Reader
	auditimport.ArtifactAccess
	lists int
	reads map[string]int
}

func (r *observedCollectionEvidence) ListExecutionEvidence(ctx context.Context, auditID, executionID string) ([]auditstore.ArtifactLink, error) {
	r.lists++
	return r.Reader.ListExecutionEvidence(ctx, auditID, executionID)
}

func (r *observedCollectionEvidence) ReadAuditExact(ctx context.Context, auditID string, ref contracts.ExactArtifact) ([]byte, error) {
	r.reads[ref.Digest]++
	return r.ArtifactAccess.ReadAuditExact(ctx, auditID, ref)
}

func (r *observedCollectionEvidence) WithClaim(claim auditstore.ControllerClaim) auditimport.ArtifactAccess {
	r.ArtifactAccess = r.ArtifactAccess.WithClaim(claim)
	return r
}

func TestSharedPostgresFinalCollectionReadsSelectedEvidenceOnce(t *testing.T) {
	f := newJudgmentFixture(t, 4, false)
	ctx := t.Context()
	storage := artifactpgtest.NewStorage()
	store := auditstoretest.NewStore(f.pool, storage)
	runs := runstore.NewPostgresStore(f.pool, storage)
	art := artifacts.NewService(storage.Repository(f.pool))
	run, err := art.Run(f.grant.RunID)
	if err != nil {
		t.Fatal(err)
	}
	source, err := run.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "selected"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("observed source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	members := []map[string]any{}
	conclusions := []auditcontract.Conclusion{}
	notes := []auditcontract.Material{}
	for index, task := range f.tasks {
		key := auditcontract.ExecutionItemKey(index)
		var selected auditcontract.EvidenceReceipt
		for note := range 8 {
			selected, err = f.service.RecordEvidence(ctx, f.grant, auditcontract.EvidenceSubmission{
				APIVersion: contracts.APIVersion, InvocationID: "invocation",
				SubmissionID: fmt.Sprintf("note-%d-%d", index, note), ItemKey: key,
				Sources: []contracts.ArtifactRef{source.Ref}, Interpretation: "Selected source observation",
				TestedConditions: []string{}, Limitations: []string{},
			})
			if err != nil {
				t.Fatal(err)
			}
			notes = append(notes, selected.Material)
		}
		input := judgmentInput(key, "Accepted final result", fmt.Sprintf("result-%d", index), 0)
		input.Items[0].Conclusion.Evidence = []auditcontract.Material{selected.Material, selected.Sources[0]}
		if _, err := f.service.RecordJudgments(ctx, f.grant, input); err != nil {
			t.Fatal(err)
		}
		conclusions = append(conclusions, input.Items[0].Conclusion)
		members = append(members, map[string]any{"item_key": key, "task_id": task.Document.TaskID, "objective": task.Document.Objective, "evidence": []auditcontract.Material{}})
	}
	digest, err := contentdigest.CanonicalJSON(f.pinned.Workflows["inspect"])
	if err != nil {
		t.Fatal(err)
	}
	trusted, err := strictjson.Canonical(map[string]any{
		"schema": "contractor.audit.context.v1", "audit_id": f.audit.AuditID,
		"run_id": f.grant.RunID, "workflow_ref": f.pinned.Program.Workflows["inspect"].Ref,
		"workflow_digest": digest, "items": members,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditcontract.Decode(trusted, nil); err != nil {
		t.Fatal(err)
	}
	scope, _ := artifacts.RunScope(f.grant.RunID)
	if _, err := art.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "context"}, artifacts.Payload{MediaType: "application/json", Data: trusted}); err != nil {
		t.Fatal(err)
	}
	final, err := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": conclusions})
	if err != nil {
		t.Fatal(err)
	}
	output, err := run.Write(ctx, contracts.ArtifactRef{Namespace: f.grant.Namespace, Name: "result"}, artifacts.Payload{MediaType: "application/json", Data: final}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, art, f.grant.RunID, "result", output.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, art, f.grant.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, f.grant.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "completed"}); err != nil {
		t.Fatal(err)
	}
	cursor, err := runs.GetRunEventCursor(ctx, f.grant.RunID)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.grant.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
	if err != nil {
		t.Fatal(err)
	}
	audit, err := store.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, storage))
	if err != nil {
		t.Fatal(err)
	}
	observed := &observedCollectionEvidence{Reader: store, ArtifactAccess: access, reads: map[string]int{}}
	collector, err := auditimport.New(store, observed, runs, observed)
	if err != nil {
		t.Fatal(err)
	}
	if changed, err := collector.Collect(ctx, f.claim, auditstore.ReconcileSnapshot{Audit: audit}, execution); err != nil || !changed {
		t.Fatalf("collection: %t %v", changed, err)
	}
	receipt, err := store.CollectionReceipt(ctx, audit.AuditID, execution.ExecutionID)
	if err != nil || receipt.Disposition != auditstore.CollectionAccepted {
		t.Fatalf("final collection was not accepted: %+v %v", receipt, err)
	}
	if observed.lists != 1 {
		t.Fatalf("evidence listed %d times", observed.lists)
	}
	for _, note := range notes {
		if observed.reads[note.Digest] != 1 {
			t.Fatalf("note %s read %d times", note.Ref.Name, observed.reads[note.Digest])
		}
	}
}
