package config

// Private reference selections parsed from Workflow YAML, including inline escalation.
type executionConfigPatch struct {
	Planner *executionSelectionPatch
	Workers *executionSelectionPatch
	Stages  map[string]stageExecutionConfigPatch
}

// stageExecutionConfigPatch is the planner and per-Agent part of one Stage's
// patch, also used when resolving an inline escalation override.
type stageExecutionConfigPatch struct {
	Planner *executionSelectionPatch
	Agents  map[string]executionSelectionPatch
}

// executionSelectionPatch selects, by reference, any of one consumer's
// ModelPolicy, LLMGatewayConfig and credential. The catalog loader builds it
// from YAML and resolves the references against its configuration set.
type executionSelectionPatch struct {
	ModelPolicy           optionalSelectionString
	SummarizerModelPolicy optionalSelectionString
	// CompactorModelPolicy applies to a compact@1 Planner or Worker (TPL-R117).
	CompactorModelPolicy optionalSelectionString
	LLMGateway           optionalSelectionString
	Credential           optionalSelectionString
}

// optionalSelectionString is one member of a patch: absent, an explicit null (which
// only a credential accepts, to clear it) or a string value.
type optionalSelectionString struct {
	Present bool
	Null    bool
	Value   string
}

func (p executionSelectionPatch) HasAny() bool {
	return p.ModelPolicy.Present || p.SummarizerModelPolicy.Present || p.CompactorModelPolicy.Present || p.LLMGateway.Present || p.Credential.Present
}
