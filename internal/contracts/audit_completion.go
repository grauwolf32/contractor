package contracts

// AuditCheckResultsV1 is an opt-in completion strategy. Declaring its wire shape
// does not advertise support or activate a Runtime strategy.
const AuditCheckResultsV1 = "audit-check-results@1"

type WorkerCompletionContract struct {
	Kind              string      `json:"kind"`
	Task              ArtifactRef `json:"task"`
	ExecutionManifest ArtifactRef `json:"executionManifest"`
	ResultArtifact    ArtifactRef `json:"resultArtifact"`
}

func (c WorkerCompletionContract) Validate() error {
	if c.Kind != AuditCheckResultsV1 {
		return invalidf("unsupported Worker completion contract")
	}
	if err := c.Task.ValidateExact(); err != nil {
		return err
	}
	if err := c.ExecutionManifest.ValidateExact(); err != nil {
		return err
	}
	if err := c.ResultArtifact.Validate(); err != nil {
		return err
	}
	if c.Task.Namespace != "inputs" || c.ExecutionManifest.Namespace != "inputs" ||
		c.Task.Name == c.ExecutionManifest.Name || c.ResultArtifact.Revision != nil ||
		c.ResultArtifact.Namespace == "inputs" {
		return invalidf("Worker completion refs require distinct exact Run inputs and a versionless output")
	}
	return nil
}

func (c WorkerCompletionContract) ValidateAllocation(namespace string, template ResolvedAgentTemplate) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if c.ResultArtifact.Namespace != namespace {
		return invalidf("Worker completion output is outside its namespace")
	}
	return ValidateAuditCompletionTemplate(template)
}

func ValidateAuditCompletionTemplate(template ResolvedAgentTemplate) error {
	if template.Summarizer != nil {
		return invalidf("Audit completion cannot select a terminal summarizer")
	}
	found := false
	for _, selection := range template.Toolsets {
		if selection.Ref.ToolsetID != "audit-results" {
			continue
		}
		if selection.Ref.Version != "2" || found || len(selection.Tools) != 2 {
			return invalidf("Audit completion requires only audit-results@2 with both tools")
		}
		tools := map[string]bool{}
		for _, name := range selection.Tools {
			tools[name] = true
		}
		if !tools["read_audit_task"] || !tools["submit_check_result"] {
			return invalidf("Audit completion requires both audit-results@2 tools")
		}
		found = true
	}
	if !found {
		return invalidf("Audit completion requires audit-results@2")
	}
	return nil
}

func CloneWorkerCompletionContract(source *WorkerCompletionContract) *WorkerCompletionContract {
	if source == nil {
		return nil
	}
	result := *source
	if source.Task.Revision != nil {
		revision := *source.Task.Revision
		result.Task.Revision = &revision
	}
	if source.ExecutionManifest.Revision != nil {
		revision := *source.ExecutionManifest.Revision
		result.ExecutionManifest.Revision = &revision
	}
	return &result
}

// ValidateWorkerCompletionSelection rejects an untrusted v2 tool selection too.
func ValidateWorkerCompletionSelection(c *WorkerCompletionContract, namespace string, template ResolvedAgentTemplate) error {
	if c != nil {
		return c.ValidateAllocation(namespace, template)
	}
	for _, toolset := range template.Toolsets {
		if toolset.Ref.ToolsetID == "audit-results" && toolset.Ref.Version == "2" {
			return invalidf("audit-results@2 requires a trusted completion contract")
		}
	}
	return nil
}

func SupportsWorkerCompletion(capabilities *RuntimeCompletionCapabilities, c *WorkerCompletionContract) bool {
	if c == nil {
		return true
	}
	if capabilities == nil {
		return false
	}
	for _, kind := range capabilities.CompletionContracts {
		if kind == c.Kind {
			return true
		}
	}
	return false
}

type RuntimeCompletionCapabilities struct {
	CompletionContracts []string `json:"completionContracts"`
}

func (c RuntimeCompletionCapabilities) Validate() error {
	if c.CompletionContracts == nil || len(c.CompletionContracts) > 1 {
		return invalidf("completionContracts must be a bounded unique capability list")
	}
	for _, kind := range c.CompletionContracts {
		if kind != AuditCheckResultsV1 {
			return invalidf("unsupported completion capability")
		}
	}
	return nil
}
