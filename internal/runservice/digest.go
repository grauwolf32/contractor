package runservice

import (
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// PublicRequestDigest is the canonical digest of a public Run creation
// request that its Idempotency-Key binds (RUN-R18): the Workflow selector,
// parameters, input refs and executionConfig, with the sorted Runtime labels
// and the normalized metadata labels, each omitted when empty. A Project Run
// also binds its Project, so one body sent to two Projects, or to a Project
// and to /v1/runs, is two different requests.
func PublicRequestDigest(params PublicCreateParams) (string, error) {
	labels, err := runtimeconfig.NormalizeRunLabels(params.RuntimeLabels)
	if err != nil {
		return "", err
	}
	parameters := params.Parameters
	if parameters == nil {
		parameters = map[string]string{}
	}
	inputs := params.Inputs
	if inputs == nil {
		inputs = map[string]contracts.ArtifactRef{}
	}
	canonical := map[string]any{
		"workflow": params.Workflow, "parameters": parameters, "artifacts": inputs,
		"executionConfig": params.ExecutionConfig.CanonicalValue(),
	}
	if len(labels) != 0 {
		canonical["runtimeLabels"] = labels
	}
	metadata, err := runstore.NormalizeRunMetadataLabels(params.MetadataLabels)
	if err != nil {
		return "", err
	}
	if len(metadata) != 0 {
		canonical["labels"] = metadata
	}
	digest, err := contentdigest.JSON(canonical)
	if err != nil || params.ProjectID == nil {
		return digest, err
	}
	return contentdigest.JSON(map[string]string{
		"sourceScope":      "project",
		"projectId":        *params.ProjectID,
		"runRequestDigest": digest,
	})
}
