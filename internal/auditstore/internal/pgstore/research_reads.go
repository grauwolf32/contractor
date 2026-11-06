package pgstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func (s *QueryStore) ListMaterials(ctx context.Context, p auditstore.MaterialListParams) (auditstore.MaterialPage, error) {
	if !pagelimit.ValidPage(p.Limit) || p.AfterOrdinal < 0 {
		return auditstore.MaterialPage{}, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.MaterialPage{}, err
	}
	if audit.State == auditstore.AuditDeleting {
		return auditstore.MaterialPage{}, auditstore.ErrNotFound
	}
	if p.AuditRevision != nil && *p.AuditRevision != audit.Revision {
		return auditstore.MaterialPage{}, auditstore.ErrConflict
	}
	rows, err := s.db.Query(ctx, listResearchMaterialsSQL, p.AuditID, p.AfterOrdinal, pagelimit.FetchRows(p.Limit))
	if err != nil {
		return auditstore.MaterialPage{}, err
	}
	defer rows.Close()
	page := auditstore.MaterialPage{Items: []auditstore.MaterialEntry{}, AuditRevision: audit.Revision}
	var candidates int
	for rows.Next() {
		var entry auditstore.MaterialEntry
		var ref json.RawMessage
		m := &entry.Material
		if err := rows.Scan(&entry.Ordinal, &entry.Key, &m.ScopeKind, &ref, &m.Digest, &m.MediaType, &m.SizeBytes, &entry.CreatedAt, &candidates); err != nil {
			return page, err
		}
		switch m.ScopeKind {
		case "audit":
			m.ScopeID = p.AuditID
		case "project":
			m.ScopeID = audit.ProjectID
		default:
			return page, errors.New("invalid stored Audit material scope")
		}
		if strictjson.Decode(ref, &m.Ref) != nil || m.Validate() != nil || entry.Key == "" || len(entry.Key) > auditstore.MaxArtifactLogicalKeyBytes || entry.CreatedAt.IsZero() {
			return page, errors.New("invalid stored Audit material")
		}
		page.Items = append(page.Items, entry)
	}
	if err := rows.Err(); err != nil {
		return page, err
	}
	rows.Close()
	if candidates > len(page.Items) || len(page.Items) > p.Limit {
		if len(page.Items) == 0 {
			return page, errors.New("Audit material exceeds page budget")
		}
		if len(page.Items) > p.Limit {
			page.Items = page.Items[:p.Limit]
		}
		ordinal := page.Items[len(page.Items)-1].Ordinal
		page.NextOrdinal = &ordinal
	}
	current, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return page, err
	}
	if current.Revision != audit.Revision {
		return page, auditstore.ErrConflict
	}
	return page, nil
}

func (s *QueryStore) ListTaskAttempts(ctx context.Context, p auditstore.TaskAttemptListParams) (auditstore.TaskAttemptPage, error) {
	if !contracts.ValidResourceID(p.TaskID) || !pagelimit.ValidPage(p.Limit) || p.AfterAttempt < 0 || p.AfterAttempt > auditprofile.MaxAuditRunAttempts {
		return auditstore.TaskAttemptPage{}, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.TaskAttemptPage{}, err
	}
	if audit.State == auditstore.AuditDeleting {
		return auditstore.TaskAttemptPage{}, auditstore.ErrNotFound
	}
	if p.AuditRevision != nil && *p.AuditRevision != audit.Revision {
		return auditstore.TaskAttemptPage{}, auditstore.ErrConflict
	}
	if _, err := s.GetTask(ctx, p.OwnerID, p.AuditID, p.TaskID); err != nil {
		return auditstore.TaskAttemptPage{}, err
	}
	rows, err := s.db.Query(ctx, listTaskAttemptsSQL, p.AuditID, p.TaskID, p.AfterAttempt, min(pagelimit.FetchRows(p.Limit), auditprofile.MaxAuditRunAttempts+1))
	if err != nil {
		return auditstore.TaskAttemptPage{}, err
	}
	defer rows.Close()
	page := auditstore.TaskAttemptPage{Items: []auditstore.TaskAttempt{}, AuditRevision: audit.Revision}
	for rows.Next() {
		var a auditstore.TaskAttempt
		if err := rows.Scan(&a.Attempt, &a.ExecutionID, &a.RunID, &a.RunDeleted, &a.Outcome, &a.Disposition, &a.State, &a.CreatedAt, &a.CollectedAt); err != nil {
			return page, err
		}
		if a.Attempt < 1 || a.Attempt > auditprofile.MaxAuditRunAttempts || !contracts.ValidResourceID(a.ExecutionID) || a.RunID != nil && !contracts.ValidResourceID(*a.RunID) || a.RunDeleted && a.RunID == nil || a.Outcome != nil && !a.Outcome.Valid() || a.Disposition != nil && !a.Disposition.Valid() || (a.State != "submitted" && a.State != "collecting" && a.State != "settled") {
			return page, errors.New("invalid stored Task attempt")
		}
		page.Items = append(page.Items, a)
	}
	if err := rows.Err(); err != nil {
		return page, err
	}
	rows.Close()
	if len(page.Items) > p.Limit {
		page.Items = page.Items[:p.Limit]
		n := page.Items[len(page.Items)-1].Attempt
		page.NextAttempt = &n
	}
	current, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return page, err
	}
	if current.Revision != audit.Revision {
		return page, auditstore.ErrConflict
	}
	return page, nil
}
