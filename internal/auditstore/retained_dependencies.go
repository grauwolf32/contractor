package auditstore

import "encoding/json"

// DependencyResultSource is an exact accepted producer item and the explicit
// retained evidence closure of its successful collection receipt. It never
// selects a role's latest result or scans artifact payload JSON.
type DependencyResultSource struct {
	ExecutionID string
	RoundID     string
	ItemID      string
	ItemAttempt int
	RunID       string
	Source      ExactArtifact
	Retained    ExactArtifact
	Evidence    []ArtifactLink
}

// ResultSourceProvenance is the shared accepted collection lineage contract.
// Both the importer writer and exact dependency reader use these JSON fields.
type ResultSourceProvenance struct {
	Schema                string          `json:"schema"`
	AuditID               string          `json:"auditId"`
	ExecutionID           string          `json:"executionId"`
	ExecutionItemID       string          `json:"executionItemId"`
	ItemID                string          `json:"itemId"`
	ItemKey               string          `json:"itemKey"`
	ItemAttempt           int             `json:"itemAttempt"`
	RunID                 string          `json:"runId"`
	WorkflowName          string          `json:"workflowName"`
	WorkflowVersion       string          `json:"workflowVersion"`
	WorkflowSchemaVersion string          `json:"workflowSchemaVersion"`
	WorkflowClosureDigest string          `json:"workflowClosureDigest"`
	ExecutionManifest     ExactArtifact   `json:"executionManifest"`
	Task                  ExactArtifact   `json:"task"`
	TaskSource            json.RawMessage `json:"taskSource"`
	SourceOutput          ExactArtifact   `json:"sourceOutput"`
}
