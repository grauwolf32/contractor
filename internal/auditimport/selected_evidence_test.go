package auditimport

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Script only immutable reads; the common decoder validates the complete data.
type selectedEvidenceReads struct {
	Reader
	ArtifactAccess
	link auditstore.ArtifactLink
	data []byte
}

func (r selectedEvidenceReads) ListExecutionEvidence(context.Context, string, string) ([]auditstore.ArtifactLink, error) {
	return []auditstore.ArtifactLink{r.link}, nil
}
func (r selectedEvidenceReads) ReadAuditExact(context.Context, string, contracts.ExactArtifact) ([]byte, error) {
	return r.data, nil
}

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
	reads := selectedEvidenceReads{link: auditstore.ArtifactLink{Artifact: noteExact}, data: noteData}
	importer := &Importer{reads: reads, artifacts: reads}
	runID := "run-1"
	for _, foreign := range []bool{false, true} {
		cited := retained
		if foreign {
			cited.Ref.Name = "unassigned"
		}
		raw, _ := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []auditcontract.Conclusion{{ItemKey: "item-00", Rationale: "Only one source was inspected", TestedConditions: []string{}, Limitations: []string{}, Evidence: []auditcontract.Material{cited, noteMaterial}}}})
		filtered, _, err := importer.taskSelectedContext(t.Context(), auditstore.Execution{AuditID: "audit-1", ExecutionID: "execution-1", RunID: &runID}, trusted, raw)
		if foreign {
			if !errors.Is(err, ErrPermanent) {
				t.Fatalf("foreign source accepted: %v", err)
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
