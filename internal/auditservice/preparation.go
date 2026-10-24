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

// ListPage reads one page of Audits and the preparation of its preparing and
// inventory Audits from one commit (ASC-R150), so a concurrent mutation or
// purge never fails the page. After that read it projects the preparation of
// the listed Audits that prepared selects, each with the same roles
// Detail returns for it at that commit; prepared may select only
// preparing or inventory Audits and runs outside the store transaction.
func (s *Service) ListPage(
	ctx context.Context, params auditstore.ListParams, prepared func(auditstore.Audit) bool,
) ([]auditstore.Audit, map[string]map[string]PreparationRoleProjection, error) {
	if prepared == nil {
		return nil, nil, errors.New("an Audit preparation selection is required")
	}
	audits, stored, err := s.reads().ListPreparedPage(ctx, params)
	if err != nil {
		return nil, nil, err
	}
	result := make(map[string]map[string]PreparationRoleProjection, len(stored))
	for _, audit := range audits {
		if !prepared(audit) {
			continue
		}
		listed, hydrated := stored[audit.AuditID]
		if !hydrated {
			return nil, nil, errors.New("the Audit preparation selection exceeds the phases that show preparation")
		}
		roles := make(map[string]PreparationRoleProjection, len(listed))
		for _, role := range listed {
			view, err := projectPreparationRole(role.PreparationRole, role.Latest, audit.State)
			if err != nil {
				return nil, nil, err
			}
			roles[role.WorkflowRole] = view
		}
		result[audit.AuditID] = roles
	}
	return audits, result, nil
}

// projectPreparationRole applies ASC-R41 to ASC-R43 to one pinned role and its
// latest attempt in an Audit in state auditState: an accepted role exposes the
// retained outputs of its Run, and a collected unaccepted attempt of a failed
// Audit reports failed. The Audit lists and the detail use it.
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
