package auditcontroller

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestResolveInputsForksTrustedExecutionManifestWithoutSelfReference(t *testing.T) {
	source := builderExact("source", "source-r1")
	task := builderExact("task", "task-r1")
	execution := builderExact("execution-manifest", "execution-r1")
	binding := config.ResolvedAuditWorkflowBinding{
		Kind: config.AuditWorkflowCheck,
		Inputs: map[string]config.AuditWorkflowInputMapping{
			"source":             {Source: config.AuditInputFromAudit, Name: "source"},
			"task":               {Source: config.AuditInputFromItemPackage},
			"execution_manifest": {Source: config.AuditInputFromExecutionManifest},
		},
	}
	baseline := auditservice.BaselineSnapshot{
		Inputs: map[string]auditstore.ExactArtifact{"source": source},
	}
	manifest := auditdomain.ExecutionItem{Inputs: []auditdomain.ExactInput{{
		Name: "source", Ref: source.Ref, Digest: source.Digest,
	}}}

	runInputs, memberInputs, err := resolveInputs(binding, baseline, manifest, task, execution)
	if err != nil {
		t.Fatal(err)
	}
	if len(runInputs) != 3 || !sameExactRef(runInputs["source"].Ref, source.Ref) ||
		!sameExactRef(runInputs["task"].Ref, task.Ref) ||
		!sameExactRef(runInputs["execution_manifest"].Ref, execution.Ref) {
		t.Fatalf("Run inputs = %+v", runInputs)
	}
	if len(memberInputs) != 1 || !sameExactRef(memberInputs[0].Ref, source.Ref) {
		t.Fatalf("execution member inputs unexpectedly include the self manifest: %+v", memberInputs)
	}
}

func TestResolveRoleInputsPinsRetainedDependencyByRoundAndLogicalName(t *testing.T) {
	baselineInput := builderExact("source", "source-r1")
	retained := builderExact("inventory", "inventory-r1")
	execution := builderExact("role-manifest", "manifest-r1")
	lookup := &fakeRoleOutputLookup{link: auditstore.ArtifactLink{Artifact: retained}}
	builder := &PinnedSubmissionBuilder{outputs: lookup}
	binding := config.ResolvedAuditWorkflowBinding{
		Inputs: map[string]config.AuditWorkflowInputMapping{
			"source":   {Source: config.AuditInputFromAudit, Name: "source"},
			"manifest": {Source: config.AuditInputFromExecutionManifest},
			"context":  {Source: config.AuditInputFromRetainedOutput, Role: "discover", Name: "inventory"},
		},
		Workflow: config.ResolvedWorkflow{Inputs: map[string]config.ArtifactSlot{
			"source": {Required: true}, "manifest": {Required: true}, "context": {Required: true},
		}},
	}
	snapshot := auditstore.ReconcileSnapshot{
		Audit: auditstore.Audit{AuditID: "audit-one"},
		Round: &auditstore.Round{Ordinal: 3},
	}

	inputs, err := builder.resolveRoleInputs(
		context.Background(), snapshot, "assess", binding,
		auditservice.BaselineSnapshot{Inputs: map[string]auditstore.ExactArtifact{"source": baselineInput}},
		execution,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 3 || inputs["source"].Digest != baselineInput.Digest ||
		inputs["manifest"].Digest != execution.Digest || inputs["context"].Digest != retained.Digest {
		t.Fatalf("resolved role inputs = %+v", inputs)
	}
	if lookup.auditID != "audit-one" ||
		lookup.logicalKey != auditstore.RoleOutputLogicalKey(3, "discover", "inventory") {
		t.Fatalf("retained lookup = (%q, %q)", lookup.auditID, lookup.logicalKey)
	}
}

type fakeRoleOutputLookup struct {
	link       auditstore.ArtifactLink
	auditID    string
	logicalKey string
}

func (f *fakeRoleOutputLookup) GetArtifactLink(
	_ context.Context, auditID, logicalKey string,
) (auditstore.ArtifactLink, error) {
	f.auditID, f.logicalKey = auditID, logicalKey
	return f.link, nil
}

func builderExact(name, revision string) auditstore.ExactArtifact {
	return auditstore.ExactArtifact{
		Ref:       contracts.ArtifactRef{Namespace: "audit-fixture", Name: name, Revision: &revision},
		Digest:    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		MediaType: "application/json", SizeBytes: 1,
	}
}
