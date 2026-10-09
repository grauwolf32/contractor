package auditservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

type PreparationRoleProjection struct {
	auditstore.PreparationRole
	RunID   *string
	Outputs map[string]auditstore.AcceptedPreparationOutput
}

// Preparation projects the bounded pinned roles, independently of item pages
// and Round history. Source Run deletion never removes accepted descriptors.
func (s *Service) Preparation(ctx context.Context, ownerID, auditID string) (map[string]PreparationRoleProjection, error) {
	store := auditstore.NewPostgresStore(s.pool)
	audit, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		return nil, err
	}
	roles, err := store.ListPreparationRoles(ctx, auditID)
	if err != nil {
		return nil, err
	}
	executions, err := store.ListPreparationExecutions(ctx, auditID)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]auditstore.Execution, len(executions))
	for _, execution := range executions {
		byID[execution.ExecutionID] = execution
	}
	result := make(map[string]PreparationRoleProjection, len(roles))
	for _, role := range roles {
		view := PreparationRoleProjection{PreparationRole: role, Outputs: map[string]auditstore.AcceptedPreparationOutput{}}
		if role.ExecutionID != nil {
			execution, exists := byID[*role.ExecutionID]
			if !exists || execution.RoleAttempt == nil {
				return nil, errors.New("stored preparation role has no matching bounded execution")
			}
			view.RunID = execution.RunID
			if role.Status == auditdomain.PreparationAccepted && execution.RunID != nil {
				for _, output := range execution.PreparationOutputs {
					view.Outputs[output.LogicalName] = auditstore.AcceptedPreparationOutput{
						ExecutionID: execution.ExecutionID, WorkflowRole: execution.WorkflowRole,
						RoleAttempt: *execution.RoleAttempt, RunID: *execution.RunID, Output: output,
					}
				}
			} else if execution.State == auditstore.ExecutionCollected && audit.State == auditstore.AuditFailed {
				view.Status = auditdomain.PreparationFailed
			}
		}
		result[role.WorkflowRole] = view
	}
	return result, nil
}
