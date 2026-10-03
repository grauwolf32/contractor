package runservice

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPrecomputedAnalysisRunInputsAreExplicitAndStrict(t *testing.T) {
	t.Parallel()

	snapshot, err := config.Load("../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	revision := func(value string) *string { return &value }
	for _, workflowRef := range []string{"openapi-from-analysis@4", "likec4-from-analysis@5"} {
		t.Run(workflowRef, func(t *testing.T) {
			workflow, workflowErr := snapshot.Workflow(workflowRef)
			if workflowErr != nil {
				t.Fatal(workflowErr)
			}
			parameters := map[string]string{"objective": "Reuse reviewed analysis"}
			inputs := map[string]contracts.ArtifactRef{
				"source":            {Namespace: "projects", Name: "source", Revision: revision("source-r1")},
				"dependency_report": {Namespace: "projects", Name: "dependencies", Revision: revision("deps-r1")},
				"project_report":    {Namespace: "projects", Name: "project", Revision: revision("project-r1")},
			}
			if err := validateWorkflowInputs(workflow, parameters, inputs); err != nil {
				t.Fatalf("complete input contract rejected: %v", err)
			}
			delete(inputs, "dependency_report")
			if err := validateWorkflowInputs(workflow, parameters, inputs); !errors.Is(err, ErrInvalid) ||
				!strings.Contains(err.Error(), "dependency_report") {
				t.Fatalf("missing dependency report error = %v", err)
			}
			if contracts.AcceptsMediaType(workflow.Inputs["dependency_report"].MediaTypes, "text/plain") ||
				!contracts.AcceptsMediaType(workflow.Inputs["dependency_report"].MediaTypes, "text/markdown") {
				t.Fatal("dependency report media-type contract is not strict")
			}
		})
	}
}
