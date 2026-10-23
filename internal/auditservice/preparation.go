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
	audit, err := s.reads().Get(ctx, ownerID, auditID)
	if err != nil {
		return nil, err
	}
	roles, err := s.reads().ListPreparationRoles(ctx, auditID)
	if err != nil {
		return nil, err
	}
	executions, err := s.reads().ListPreparationExecutions(ctx, auditID)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]auditstore.Execution, len(executions))
	for _, execution := range executions {
		byID[execution.ExecutionID] = execution
	}
	result := make(map[string]PreparationRoleProjection, len(roles))
	for _, role := range roles {
		var latest *auditstore.PreparationAttempt
		if role.ExecutionID != nil {
			execution, exists := byID[*role.ExecutionID]
			if !exists || execution.RoleAttempt == nil {
				return nil, errors.New("stored preparation role has no matching bounded execution")
			}
			latest = &auditstore.PreparationAttempt{
				ExecutionID: execution.ExecutionID, RoleAttempt: *execution.RoleAttempt, RunID: execution.RunID,
				State: execution.State, Outputs: execution.PreparationOutputs,
			}
		}
		view, err := projectPreparationRole(role, latest, audit.State)
		if err != nil {
			return nil, err
		}
		result[role.WorkflowRole] = view
	}
	if err := s.checkPageRevision(ctx, ownerID, auditID, audit.Revision); err != nil {
		return nil, err
	}
	return result, nil
}

// PreparationPage projects the preparation of the Audits of one list page
// with a single store read, whatever the number of Audits, each fenced by the
// revision the page listed (ASC-R150). A listed Audit that is gone or has
// another revision returns auditstore.ErrConflict (ASC-R149). The result maps
// every listed Audit to the same roles Preparation returns for it.
func (s *Service) PreparationPage(
	ctx context.Context, ownerID string, audits []auditstore.Audit,
) (map[string]map[string]PreparationRoleProjection, error) {
	listed := make([]auditstore.ListedAudit, len(audits))
	for index, audit := range audits {
		listed[index] = auditstore.ListedAudit{AuditID: audit.AuditID, Revision: audit.Revision}
	}
	stored, err := s.reads().ListPreparationPage(ctx, ownerID, listed)
	if err != nil {
		return nil, err
	}
	result := make(map[string]map[string]PreparationRoleProjection, len(audits))
	for _, audit := range audits {
		roles := make(map[string]PreparationRoleProjection, len(stored[audit.AuditID]))
		for _, role := range stored[audit.AuditID] {
			view, err := projectPreparationRole(role.PreparationRole, role.Latest, audit.State)
			if err != nil {
				return nil, err
			}
			roles[role.WorkflowRole] = view
		}
		result[audit.AuditID] = roles
	}
	return result, nil
}

// projectPreparationRole applies ASC-R41 to ASC-R43 to one pinned role and its
// latest attempt in an Audit in state auditState: an accepted role exposes the
// retained outputs of its Run, and a collected unaccepted attempt of a failed
// Audit reports failed. Both preparation reads use it.
func projectPreparationRole(
	role auditstore.PreparationRole, latest *auditstore.PreparationAttempt, auditState auditstore.AuditState,
) (PreparationRoleProjection, error) {
	view := PreparationRoleProjection{PreparationRole: role, Outputs: map[string]auditstore.AcceptedPreparationOutput{}}
	if role.ExecutionID == nil {
		return view, nil
	}
	if latest == nil || latest.ExecutionID != *role.ExecutionID {
		return PreparationRoleProjection{}, errors.New("stored preparation role has no matching bounded execution")
	}
	view.RunID = latest.RunID
	if role.Status == auditdomain.PreparationAccepted && latest.RunID != nil {
		for _, output := range latest.Outputs {
			view.Outputs[output.LogicalName] = auditstore.AcceptedPreparationOutput{
				ExecutionID: latest.ExecutionID, WorkflowRole: role.WorkflowRole,
				RoleAttempt: latest.RoleAttempt, RunID: *latest.RunID, Output: output,
			}
		}
	} else if latest.State == auditstore.ExecutionCollected && auditState == auditstore.AuditFailed {
		view.Status = auditdomain.PreparationFailed
	}
	return view, nil
}
