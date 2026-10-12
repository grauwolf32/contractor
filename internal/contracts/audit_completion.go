package contracts

// AuditCheckResultsV1 is an opt-in completion strategy. Declaring its wire shape
// does not advertise support or activate a Runtime strategy.
const AuditCheckResultsV1 = "audit-check-results@1"

type WorkerCompletionContract struct {
	Kind              string      `json:"kind" strictjson:"required"`
	Task              ArtifactRef `json:"task" strictjson:"required"`
	ExecutionManifest ArtifactRef `json:"executionManifest" strictjson:"required"`
	ResultArtifact    ArtifactRef `json:"resultArtifact" strictjson:"required"`
}

func (c WorkerCompletionContract) Validate() error {
	if c.Kind != AuditCheckResultsV1 {
		return Invalidf("unsupported Worker completion contract")
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
		return Invalidf("Worker completion refs require distinct exact Run inputs and a versionless output")
	}
	return nil
}

// ValidateAllocation checks the contract as an AllocationSpec carries it: its
// output lies in the allocation namespace. Which AgentTemplate can complete
// under it is the audit-results@2 Toolset's rule
// (resolved.ValidateWorkerCompletionSelection).
func (c WorkerCompletionContract) ValidateAllocation(namespace string) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if c.ResultArtifact.Namespace != namespace {
		return Invalidf("Worker completion output is outside its namespace")
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
	CompletionContracts []string `json:"completionContracts" strictjson:"required"`
}

func (c RuntimeCompletionCapabilities) Validate() error {
	if c.CompletionContracts == nil || len(c.CompletionContracts) > 1 {
		return Invalidf("completionContracts must be a bounded unique capability list")
	}
	for _, kind := range c.CompletionContracts {
		if kind != AuditCheckResultsV1 {
			return Invalidf("unsupported completion capability")
		}
	}
	return nil
}
