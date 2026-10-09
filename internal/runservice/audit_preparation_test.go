package runservice

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func exactTestRef(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func TestPreparationRunRequiresTheCompletePinnedSubmission(t *testing.T) {
	revision := "r1"
	input := auditstore.ExactArtifact{Ref: exactTestRef("sources", "source", revision), Digest: testDigest("1"), MediaType: "application/zip", SizeBytes: 100}
	manifest := auditstore.ExactArtifact{Ref: exactTestRef("audit-manifests", "empty", revision), Digest: testDigest("2")}
	workflow := config.ResolvedWorkflow{}
	binding, _ := json.Marshal(config.ResolvedAuditWorkflowBinding{Kind: config.AuditWorkflowPrepare, Workflow: workflow})
	intent := auditstore.RunCreationIntent{WorkflowBinding: binding, Execution: auditstore.Execution{
		ExecutionID: "execution", AuditID: "audit", Role: auditstore.ExecutionPrepare, Manifest: manifest, RequestDigest: testDigest("3"),
		Preparation: &auditstore.PreparationSnapshot{Inputs: map[string]auditstore.ExactArtifact{"source": input}, Parameters: map[string]string{"objective": "review"}},
	}}
	params := AuditCreateParams{ExecutionID: "execution", RequestDigest: testDigest("3"), ExecutionManifest: manifest, Workflow: workflow,
		Inputs: map[string]auditstore.ExactArtifact{"source": input}, Parameters: map[string]string{"objective": "review"}}
	params.Claim.AuditID = "audit"
	if err := validateAuditIntent(intent, params); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		change func(*AuditCreateParams)
	}{
		{"missing-input", func(p *AuditCreateParams) { p.Inputs = map[string]auditstore.ExactArtifact{} }},
		{"wrong-revision", func(p *AuditCreateParams) {
			value := input
			value.Ref = exactTestRef("sources", "source", "r2")
			p.Inputs = map[string]auditstore.ExactArtifact{"source": value}
		}},
		{"wrong-media", func(p *AuditCreateParams) {
			value := input
			value.MediaType = "text/plain"
			p.Inputs = map[string]auditstore.ExactArtifact{"source": value}
		}},
		{"wrong-size", func(p *AuditCreateParams) {
			value := input
			value.SizeBytes++
			p.Inputs = map[string]auditstore.ExactArtifact{"source": value}
		}},
		{"wrong-parameters", func(p *AuditCreateParams) { p.Parameters = map[string]string{"objective": "changed"} }},
		{"wrong-workflow", func(p *AuditCreateParams) { p.Workflow.Ref.Name = "changed" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := params
			tc.change(&changed)
			if err := validateAuditIntent(intent, changed); !errors.Is(err, auditstore.ErrConflict) {
				t.Fatalf("changed preparation authority accepted: %v", err)
			}
		})
	}
}
