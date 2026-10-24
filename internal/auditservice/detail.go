package auditservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// AuditDetail is the detail projection of one Audit (ASC-R149). Preparation
// is nil unless the detail hydrates it; GeneratedInventory is set only once
// the hydrated Audit is in the rounds phase and not deleting (ASC-R146).
type AuditDetail struct {
	Audit              auditstore.Audit
	Preparation        map[string]PreparationRoleProjection
	GeneratedInventory *GeneratedInventoryProjection
}

// StartedDetail is a start response (AUD-R226): the detail of the started
// Audit, its current Round, nil while it has none, and only that Round's
// items.
type StartedDetail struct {
	AuditDetail
	Round *auditstore.Round
	Items []auditstore.Item
}

// Detail returns the detail of owner ownerID's Audit, given audit, the state
// of it that a read or a committed mutation returned (ASC-R149). prepared
// reports whether the detail of audit hydrates preparation; an Audit cannot
// return to not-started and its pinned profile is immutable once it started,
// so a later state of it hydrates too. Detail calls prepared once, before any
// read. An Audit it does not select is its own detail, without another read.
// A selected one is read again, with its preparation, initial-inventory link
// and first Round, in one read-only snapshot, so the detail, its revision and
// its ETag come from one commit at or after audit's. A write that commits
// meanwhile never fails the read: the detail is the Audit as of a commit
// before or after it, as a later read or an Idempotency-Key replay of the
// mutation would also return it.
func (s *Service) Detail(
	ctx context.Context, ownerID string, audit auditstore.Audit, prepared func(auditstore.Audit) bool,
) (AuditDetail, error) {
	if prepared == nil {
		return AuditDetail{}, errors.New("an Audit preparation selection is required")
	}
	if !prepared(audit) {
		return AuditDetail{Audit: audit}, nil
	}
	stored, err := s.reads().GetPreparedDetail(ctx, ownerID, audit.AuditID, false)
	if err != nil {
		return AuditDetail{}, err
	}
	return s.preparedDetail(ctx, stored)
}

// StartedDetail returns the start response of started, the projection of a
// start or of its replay, as Detail returns the Audit's detail. A selected
// Audit's current Round and its items come from the snapshot of its detail
// too, so the Round is the current Round of the Audit the response carries,
// whatever Round began after started was read.
func (s *Service) StartedDetail(
	ctx context.Context, ownerID string, started StartedAudit, prepared func(auditstore.Audit) bool,
) (StartedDetail, error) {
	if prepared == nil {
		return StartedDetail{}, errors.New("an Audit preparation selection is required")
	}
	if !prepared(started.Audit) {
		result := StartedDetail{AuditDetail: AuditDetail{Audit: started.Audit}, Items: started.Items}
		if started.Audit.CurrentRoundID != nil {
			round := started.Round
			result.Round = &round
		}
		return result, nil
	}
	stored, err := s.reads().GetPreparedDetail(ctx, ownerID, started.Audit.AuditID, true)
	if err != nil {
		return StartedDetail{}, err
	}
	detail, err := s.preparedDetail(ctx, stored)
	if err != nil {
		return StartedDetail{}, err
	}
	items := stored.CurrentItems
	if items == nil {
		items = []auditstore.Item{}
	}
	return StartedDetail{AuditDetail: detail, Round: stored.CurrentRound, Items: items}, nil
}

// preparedDetail projects the rows of one Audit's detail snapshot. It runs
// after the snapshot committed: the only read left is the exact
// initial-inventory snapshot, which is immutable.
func (s *Service) preparedDetail(ctx context.Context, stored auditstore.PreparedDetail) (AuditDetail, error) {
	if stored.Audit.Phase == auditdomain.AuditPhaseNotStarted {
		return AuditDetail{Audit: stored.Audit}, nil
	}
	result := AuditDetail{Audit: stored.Audit, Preparation: make(map[string]PreparationRoleProjection, len(stored.Roles))}
	for _, role := range stored.Roles {
		view, err := projectPreparationRole(role.PreparationRole, role.Latest, stored.Audit.State)
		if err != nil {
			return AuditDetail{}, err
		}
		result.Preparation[role.WorkflowRole] = view
	}
	if stored.InventoryLink == nil {
		return result, nil
	}
	if stored.FirstRound == nil {
		return AuditDetail{}, errors.New("an Audit detail with generated inventory has no first Round")
	}
	inventory, err := s.generatedInventory(ctx, stored.Audit, *stored.InventoryLink, *stored.FirstRound)
	if err != nil {
		return AuditDetail{}, err
	}
	result.GeneratedInventory = inventory
	return result, nil
}
