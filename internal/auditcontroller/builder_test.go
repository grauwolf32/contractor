package auditcontroller

import (
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

func builderExact(name, revision string) auditstore.ExactArtifact {
	return auditstore.ExactArtifact{
		Ref:       contracts.ArtifactRef{Namespace: "audit-fixture", Name: name, Revision: &revision},
		Digest:    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		MediaType: "application/json", SizeBytes: 1,
	}
}
