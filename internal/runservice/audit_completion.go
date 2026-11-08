package runservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Completion ownership is derived solely from the claim-locked common pin.
func prepareAuditCompletion(binding auditcontract.ProgramWorkflow, params auditRunCreate) (*runstore.AuditCompletionSnapshot, error) {
	target, err := auditprofile.ResolveCompletionTarget(params.Workflow, binding)
	if err != nil {
		return nil, fmt.Errorf("%w: common completion target is invalid", ErrInvalid)
	}
	if target == nil {
		return nil, nil
	}
	return &runstore.AuditCompletionSnapshot{Stage: target.Stage, Agent: target.Agent, Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1, Context: contracts.ArtifactRef{Namespace: "inputs", Name: target.ContextSlot}, ResultArtifact: target.Result}}, nil
}
