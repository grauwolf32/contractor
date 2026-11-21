package resolved

import "github.com/grauwolf32/contractor/internal/contracts"

// The audit-results@2 Toolset records Audit check results under a trusted
// Worker completion contract. Its descriptor and the completion rules below
// share these names, so the rules follow the descriptor.
const (
	AuditResultsToolsetID      = contracts.AuditResultsToolsetID
	AuditResultsToolsetVersion = contracts.AuditResultsToolsetVersion
	ReadAuditTaskTool          = "read_audit_task"
	SubmitCheckResultTool      = "submit_check_result"
	SubmitCheckResultsTool     = "submit_check_results"
)

// ValidateAuditCompletionTemplate checks that an AgentTemplate can complete
// Audit checks: it selects audit-results@2 once, with all of its tools
// (AWC-R11), and no effective terminal summarizer (AWC-R12). compact@1
// wholly replaces a template summarizer without changing the template.
func ValidateAuditCompletionTemplate(template contracts.ResolvedAgentTemplate, context contracts.WorkerContextManagement) error {
	if context.Strategy == contracts.ContextStrategyTerminal || (template.Summarizer != nil && !context.IsCompact()) {
		return contracts.Invalidf("Audit completion cannot select a terminal summarizer")
	}
	found := false
	for _, selection := range template.Toolsets {
		if selection.Ref.ToolsetID != AuditResultsToolsetID {
			continue
		}
		if selection.Ref.Version != AuditResultsToolsetVersion || found || len(selection.Tools) != 3 {
			return contracts.Invalidf("Audit completion requires only audit-results@2 with all tools")
		}
		tools := map[string]bool{}
		for _, name := range selection.Tools {
			tools[name] = true
		}
		if !tools[ReadAuditTaskTool] || !tools[SubmitCheckResultTool] || !tools[SubmitCheckResultsTool] {
			return contracts.Invalidf("Audit completion requires all audit-results@2 tools")
		}
		found = true
	}
	if !found {
		return contracts.Invalidf("Audit completion requires audit-results@2")
	}
	return nil
}

// ValidateWorkerCompletionSelection checks the completion choice of one
// allocation: a pinned contract must fit the allocation namespace (the wire
// check of the contract) and an AgentTemplate that can complete Audit checks,
// and audit-results@2 is never selected without a trusted contract (AWC-R18).
func ValidateWorkerCompletionSelection(
	contract *contracts.WorkerCompletionContract,
	namespace string,
	template contracts.ResolvedAgentTemplate,
	context contracts.WorkerContextManagement,
) error {
	if contract != nil {
		if err := contract.ValidateAllocation(namespace); err != nil {
			return err
		}
		return ValidateAuditCompletionTemplate(template, context)
	}
	for _, toolset := range template.Toolsets {
		if toolset.Ref.ToolsetID == AuditResultsToolsetID && toolset.Ref.Version == AuditResultsToolsetVersion {
			return contracts.Invalidf("audit-results@2 requires a trusted completion contract")
		}
	}
	return nil
}
