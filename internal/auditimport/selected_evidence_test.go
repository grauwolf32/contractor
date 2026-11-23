package auditimport

import (
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestResultContextDoesNotCountUncitedAssignedMaterials(t *testing.T) {
	// Exercise the real common decoder with its largest valid assignment list.
	// No store mutation is emulated here; PG collection is tested in runservice.
	material := auditcontract.Material{ScopeKind: "run", ScopeID: "run-1", Digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", MediaType: "text/plain", SizeBytes: 3}
	revision := "revision-1"
	material.Ref.Namespace, material.Ref.Revision = "inputs", &revision
	assigned := []auditcontract.Material{}
	for index := range 256 {
		source := material
		source.Ref.Name = fmt.Sprintf("source-%03d", index)
		assigned = append(assigned, source)
	}
	trusted, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.context.v1", "audit_id": "audit-1", "run_id": "run-1", "workflow_ref": "inspect@1", "workflow_digest": material.Digest, "items": []any{map[string]any{"item_key": "item-00", "task_id": "task-1", "objective": "Inspect selected sources", "evidence": assigned}}})
	if _, err := auditcontract.Decode(trusted, nil); err != nil {
		t.Fatal(err)
	}
	retained := assigned[0]
	retained.ScopeKind, retained.ScopeID = "audit", "audit-1"
	retained.Ref.Namespace, retained.Ref.Name = "audit-sources", "retained"
	note := auditcontract.EvidenceRecord{
		Schema: "contractor.audit.evidence.v1", RequestDigest: material.Digest,
		Origin:  auditcontract.EvidenceOrigin{AuditID: "audit-1", TaskID: "task-1", RunID: "run-1", ExecutionID: "execution-1", AllocationID: "allocation-1", StageExecutionID: "stage-1", LogicalAgentName: "worker", InvocationID: "invocation-1", SubmissionID: "submission-1"},
		Sources: []auditcontract.Material{retained}, SelectedSources: []auditcontract.Material{assigned[0]},
		Interpretation: "A selected source for this check", TestedConditions: []string{}, Limitations: []string{},
	}
	if err := note.Validate(); err != nil {
		t.Fatal(err)
	}
	noteData, err := strictjson.Canonical(note)
	if err != nil {
		t.Fatal(err)
	}
	noteRef := retained.Ref
	noteRef.Name = "note"
	noteExact := contracts.ExactContent(noteRef, "application/json", noteData)
	noteMaterial := auditcontract.Material{ScopeKind: "audit", ScopeID: "audit-1", Ref: noteRef, Digest: noteExact.Digest, MediaType: noteExact.MediaType, SizeBytes: noteExact.SizeBytes}
	for _, scenario := range []string{"accepted", "foreign-source", "foreign-task", "foreign-hypothesis"} {
		evidence := selectedTaskEvidence{materials: []auditcontract.Material{retained, noteMaterial}}
		taskID := "task-1"
		if scenario == "foreign-task" {
			taskID = "unassigned"
		}
		if scenario == "foreign-hypothesis" {
			evidence.hypothesis = &auditcontract.HypothesisRef{ID: "unassigned", Revision: 1}
		}
		partials := partialTaskResults{evidence: map[string]selectedTaskEvidence{taskID: evidence}}
		cited := retained
		if scenario == "foreign-source" {
			cited.Ref.Name = "unassigned"
		}
		raw, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []auditcontract.Conclusion{{ItemKey: "item-00", Rationale: "Only one source was inspected", TestedConditions: []string{}, Limitations: []string{}, Evidence: []auditcontract.Material{cited, noteMaterial}}}})
		filtered, _, err := partials.taskSelectedContext(trusted, raw)
		if scenario != "accepted" {
			if !errors.Is(err, ErrPermanent) {
				t.Fatalf("%s accepted: %v", scenario, err)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, err := auditcontract.Decode(raw, filtered); err != nil {
			t.Fatalf("authorized result rejected: %v", err)
		}
		var value map[string]any
		if err := strictjson.Decode(filtered, &value); err != nil {
			t.Fatal(err)
		}
		if len(value["items"].([]any)[0].(map[string]any)["evidence"].([]any)) != 2 {
			t.Fatal("uncited inputs entered result membership")
		}
	}
}
